"""Run DynaGuide rollout evaluation and dataset/video generation.

This script executes rollouts using a diffusion policy and dynamics guidance model,
supports optional video recording, HDF5 rollout export, and environment setup overrides.

Example:
    python dyna_scripts/run_dynaguide.py --dp_model model.pth --dynamics_model dyn.pth --n_rollouts 20 --dataset_path rollouts.hdf5 --output_folder results
"""

import argparse
import os
import sys

# Resolve the absolute path of the project root directory (one level up from dyna_scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "dyna_scripts" else SCRIPT_DIR

# Ensure the root project folder is searched first for local modules like 'core'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Handle local repository overrides for robosuite and mimicgen cleanly
for local_repo in ["robosuite", "mimicgen"]:
    repo_path = os.path.join(PROJECT_ROOT, local_repo)
    if os.path.exists(repo_path) and not os.path.exists(os.path.join(repo_path, "__init__.py")):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

import json
import h5py
import imageio
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import math

import shutil 
import torchvision 
import torch
import pickle 

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS
import io 
import random 

import cv2 
import matplotlib.pyplot as plt 

# these are the core modules that create the dynaguide behavior 
from core.dynamics_models import FinalStatePredictionDino
from core.embedder_datasets import MultiviewDataset
from core.calvin_utils import generate_reset_state, check_state_difference
from core.dynaguide import calculate_classifier_guidance, calculate_position_guidance

CLUSTER_ORDER = ["Right (45-135°)", "Back (135-225°)", "Left (225-315°)", "Front (315-45°)"]


def get_orientation_cluster(yaw_deg):
    """Map a yaw angle (degrees) to one of 4 insertion-direction clusters (robot-relative)."""
    y = yaw_deg % 360
    if 45 <= y < 135:
        return "Right (45-135°)"
    elif 135 <= y < 225:
        return "Back (135-225°)"
    elif 225 <= y < 315:
        return "Left (225-315°)"
    else:
        return "Front (315-45°)"


def get_img_from_fig(fig, dpi=80):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 


def visualize_guidance_process(model, actions_list, obs, target_embeddings, alpha = 20):
    """
    This function visualizes the reconstruction throughout the denoising process and also prints the latent distance for reference on the image 
    """
    # assemble proprio
    if "proprio" not in obs:
        if all(k in obs for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]):
            obs = deepcopy(obs)  # Avoid modifying the original loop reference
            obs["proprio"] = np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"]
            ], axis=-1)

    SKIP_RATE = 1 # skip every 5 
    resizer = torchvision.transforms.Resize((128, 128))

    mod_obs = TensorUtils.to_tensor(obs)
    mod_obs = TensorUtils.to_batch(mod_obs)
    mod_obs = TensorUtils.to_device(mod_obs, "cuda")
    mod_obs = TensorUtils.to_float(mod_obs)
    if len(mod_obs[MAIN_CAMERA].shape) == 4:
        mod_obs[MAIN_CAMERA] = torch.unsqueeze(mod_obs[MAIN_CAMERA], dim = 1) # for policies that don't have framestacking; in dp this is done automatically 
    if len(mod_obs["proprio"].shape) == 2:
        mod_obs["proprio"] = torch.unsqueeze(mod_obs["proprio"], dim = 1) # for policies that don't have framestacking; in dp this is done automatically 
    
    fed_state = {MAIN_CAMERA: mod_obs[MAIN_CAMERA][:, -1] * 255}
    fed_state["proprio"] = mod_obs["proprio"][:, -1]

    # required_padding = 5 - len(actions_list) % 5 if len(actions_list) % 5 != 0 else 0 
    required_padding = 5 - math.ceil(len(actions_list) / SKIP_RATE) % 5 if math.ceil(len(actions_list) / SKIP_RATE) % 5 != 0 else 0 
    pad = np.zeros((128, 128, 3), dtype = np.uint8)
    predicted_image_list = list()

    # Gather actions to run in a single batch pass
    actions_to_feed = [actions_list[i] for i in range(0, len(actions_list), SKIP_RATE)]

    if len(actions_to_feed) > 0:
        # Concatenate along the existing batch dimension (dim 0) to maintain a 3D tensor [N, T, D]
        if isinstance(actions_to_feed[0], np.ndarray):
            if actions_to_feed[0].ndim == 3:
                batch_actions = torch.tensor(np.concatenate(actions_to_feed, axis=0), dtype=torch.float32, device="cuda")
            else:
                batch_actions = torch.tensor(np.array(actions_to_feed), dtype=torch.float32, device="cuda")
        else:
            if actions_to_feed[0].dim() == 3:
                batch_actions = torch.cat(actions_to_feed, dim=0).to("cuda").float()
            else:
                batch_actions = torch.stack(actions_to_feed).to("cuda").float()
            
        # Match the batch dimension for our state observations 
        batch_state = {k: v.repeat(batch_actions.shape[0], *([1] * (len(v.shape) - 1))) for k, v in fed_state.items()}
        
        with torch.no_grad():
            s_end_embeddings, reco_images = model(batch_state, batch_actions)
            reco_images = torch.clip(resizer(reco_images), 0, 1)
            
        s_end_embeddings = s_end_embeddings.flatten(start_dim=1)
        if target_embeddings is not None:
            s_norm = torch.cdist(target_embeddings, s_end_embeddings, p=2.0)
            sa_average_norms = -torch.logsumexp(-s_norm / alpha, dim=0).detach().cpu().numpy()

        # Post-processing loop for image transformation and text overlay
        for idx in range(len(actions_to_feed)):
            reco_image = np.transpose(np.clip(reco_images[idx].detach().cpu().numpy(), 0, 1), (1, 2, 0)).copy() * 255
            reco_image = reco_image.astype(np.uint8)
            if target_embeddings is not None:
                sa_average_norm = sa_average_norms[idx]
                reco_image = cv2.putText(reco_image, str(int(sa_average_norm.item())), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            predicted_image_list.append(reco_image.copy())

    grid_list = list()
    row_list = list() 
    for i in range(len(predicted_image_list)):
        if i % 5 == 0 and i > 0:
            grid_list.append(np.concatenate(row_list, axis = 1))
            row_list = list()
        row_list.append(predicted_image_list[i])
    for i in range(required_padding):
        row_list.append(pad)
    grid_list.append(np.concatenate(row_list, axis = 1)) # final row might have black padding 
    collage = np.concatenate(grid_list, axis = 0)
    return collage 


def rollout(policy, env, horizon, video_writer=None, video_skip=5, return_obs=False, camera_names=None,
            classifier_grad = None, reset_state = None, model = None, exp_dir = None, rollout_count = None, good_embeddings = None, bad_embeddings = None,
            setup_config = None, save_frames = False, goal_dir = None, ss = 4, alpha = 20, initial_positions = None, render_denoising = False,
            use_sequential_poses = False, pose_index = 0, save_images = False):
    """
    This is the main function that runs DynaGuide. Notable arguments:
    - classifier_grad: the dynaguide function returned by calculate_classifier_guidance 
    - reset_state: a fixed reset state for experiments where you want to remove environment randomness ; not used 
    - model: base policy
    - good_embeddings: embeddings of the desired states (if any), used for visualization 
    - bad_embeddings: embeddings of the undesired states (if any), used for visualization
    - setup_config: environment configs relevant for the experiment 
    - ss: number of times to run stochastic sampling 
    - alpha: how much to divide the l2 distance, corresponding to sigma in the paper 
    - initial_positions: a list of initial robot configurations for resetting 
    - goal_dir: place to retrieve goal images; if none, we are not goal conditioning 
    """
    rollout_timestamp = time.time()
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)

    policy.start_episode()

    obs = env.reset()   

    # for goal conditioning baseline 
    if goal_dir is not None:
        env.set_goal(im_paths = goal_dir)
   
    # this is for calvin environment     
    special_state, articulated_binaries = generate_reset_state(sim_hold = setup_config["env_setup"]) # and keep the switch off 

    if initial_positions is not None:
        if use_sequential_poses:
            state_to_reset = {"states": np.array(initial_positions[pose_index])}
        else:
            state_to_reset = {"states": np.array(random.choice(initial_positions))}
    else:
        state_to_reset = special_state

    # Wrap raw list or array vectors into the dictionary format robomimic expects
    if not isinstance(state_to_reset, dict):
        state_to_reset = {"states": np.array(state_to_reset)}
        
    obs = env.reset_to(state_to_reset)

    state_dict = env.get_state()

    # FOR DEBUGGING ONLY
    # articulated_binaries = articulated_binaries_from_start_state(state_dict) # if you don't reset to special state you need to get the right binaries 

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    got_exception = False
    success = env.is_success()["task"]
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    start_state = obs.copy()
    collage = None
    t_14_start = time.time()
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):
            t1 = time.time()
            if goal_dir is not None: # goal conditioning 
                goal_dict = env.get_goal()
                act = policy(ob=obs, guidance_function = classifier_grad, goal = goal_dict, guidance_type = "diffusion", ss = ss)
            else:
                act = policy(ob=obs, guidance_function = classifier_grad, guidance_type = "diffusion", ss = ss) # this is where the guidance happens 

            t2 = time.time()

            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            if render_denoising and step_i % 14 == 0: # after every resampling, update the visuals 
                denoising_list = policy.policy.corrections_list 
                guidance_list = policy.policy.guidance_list
                print("Guidance norms: ", [np.linalg.norm(guidance.detach().cpu().numpy()).item() for guidance in guidance_list])
                target_embeddings = bad_embeddings if good_embeddings is None else good_embeddings
                collage = visualize_guidance_process(model, denoising_list, obs, target_embeddings, alpha = alpha)
         
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        render_cam = cam_name.replace("_image", "")
                        img = env.render(mode="rgb_array", height=256, width=256, camera_name=render_cam)
                        video_img.append(img)
                    video_img = np.concatenate(video_img, axis=0) # concatenate horizontally
                    video_img = cv2.resize(video_img, (128 * 5, 128 * 5))

                    # Comment this block, if you dont want the visual lables
                    if success:
                        text = "SUCCESS"
                        color = (0, 255, 0) if success else (255, 0, 0)  # Green if True, Red if False
                        cv2.putText(video_img, text, (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
                    # ------------------------------

                    if collage is not None:
                        video_img = np.concatenate((collage, video_img), axis = 0)
                    video_writer.append_data(video_img)
                
                # track ongoing benchmark progress
                if step_i % 14 == 0:
                    print(f"\n>>> [Active Progress] Rollout: {rollout_count + 1} | Environment Step: {step_i} / {horizon}")
                    if save_frames:
                        plt.imsave(exp_dir + f"/{rollout_count}_{step_i}.png", video_img)
                
                video_count += 1

            if step_i > 0 and step_i % 14 == 0:
                print(f"[Timing] Environment steps {step_i - 14} to {step_i} took: {time.time() - t_14_start:.4f} seconds")
                t_14_start = time.time()


            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            if isinstance(state_dict, dict) and "states" in state_dict:
                traj["states"].append(state_dict["states"])
            else:
                traj["states"].append(0) # Fallback baseline if environment lacks states
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                
                # Create copies so we don't mutilate the live environment dictionary keys
                obs_to_save = deepcopy(obs)
                next_obs_to_save = deepcopy(next_obs)
                
                if not save_images and not return_obs:
                    for cam_name in camera_names:
                        obs_to_save.pop(cam_name, None)
                        next_obs_to_save.pop(cam_name, None)
                
                obs_to_save.pop("third_person", None)
                obs_to_save.pop("eye_in_hand", None)
                next_obs_to_save.pop("third_person", None)
                next_obs_to_save.pop("eye_in_hand", None)

                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs_to_save))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs_to_save))

            # break if done or if success
            if done or success:
                # always capture the last frame
                video_img = []
                for cam_name in camera_names:
                    render_cam = cam_name.replace("_image", "")
                    img = env.render(mode="rgb_array", height=128 * 5, width=128 * 5, camera_name=render_cam)
                    video_img.append(img)
                video_img = np.concatenate(video_img, axis=0) # concatenate horizontally
                video_img = cv2.resize(video_img, (128 * 5, 128 * 5))

                # --- ADD LIVE VISUAL LABEL ---
                text = "SUCCESS" if success else "FAIL"
                color = (0, 255, 0) if success else (255, 0, 0)
                cv2.putText(video_img, text, (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

                # --- OVERLAY ORIENTATION CLUSTER on success ---
                if success:
                    rs_env = _unwrap_env(env)
                    if hasattr(rs_env, 'assembled_nut_yaw'):
                        yaw = rs_env.assembled_nut_yaw()
                        if yaw is not None:
                            cluster = get_orientation_cluster(yaw)
                            cv2.putText(video_img, f"Yaw: {yaw:.1f}deg", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(video_img, f"Dir: {cluster}", (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                # ------------------------------

                if collage is not None:
                        video_img = np.concatenate((collage, video_img), axis = 0)
                video_writer.append_data(video_img)
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
        got_exception = True

    stats = dict(
        Return=total_reward,
        Horizon=(step_i + 1),
        Success_Rate=float(success),
        Exception_Rate=float(got_exception),
        time=(time.time() - rollout_timestamp),
    )

    # Yaw/orientation tracking for all NutAssembly environments
    rs_env = _unwrap_env(env)
    if hasattr(rs_env, 'assembled_nut_yaw'):
        stats['assembly_yaw_deg'] = rs_env.assembled_nut_yaw()  # float | None
    if hasattr(rs_env, 'which_nut_assembled'):
        stats['assembled_nut'] = rs_env.which_nut_assembled()   # 'SquareNut' | 'RedSquareNut' | None

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])


    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
                # HACK for removing frame stacking (Made safe for Robosuite)
                if (k == "obs" or k == "next_obs") and "states" in traj[k] and len(traj[k]["states"][0].shape) > 1:
                    traj[k][kp] = traj[k][kp][:, -1] 
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def _unwrap_env(env):
    """Peel robomimic wrapper layers to reach the underlying robosuite environment."""
    e = env
    while hasattr(e, 'env'):
        e = e.env
    return e


_OOD_ENV_MAP = {
    "none":       None,
    "red_nut":    "NutAssemblyRedSquare",
    "dual_red":   "NutAssemblyDualSquare",
    "dual_gold":  "NutAssemblyDualSquare",
}

_OOD_CONFIG_MAP = {
    "none":       None,
    "red_nut":    "ood_configs/ood_red_nut.json",
    "dual_red":   "ood_configs/dual_pick_red.json",
    "dual_gold":  "ood_configs/dual_pick_gold.json",
}

_OOD_COLLECT_HINT = {
    "red_nut": (
        "Collect guidance data with:\n"
        "  python robomimic/robomimic/scripts/run_trained_agent.py "
        "--agent <checkpoint> --env NutAssemblyRedSquare --n_rollouts 20 "
        "--dataset_path dataset/ood_red_nut/rollouts.hdf5 --dataset_obs\n"
        "Then create pos/neg snapshots with helper_scripts/create_guidance_dataset.py"
    ),
    "dual_red": (
        "Collect guidance data with:\n"
        "  python robomimic/robomimic/scripts/run_trained_agent.py "
        "--agent <checkpoint> --env NutAssemblyDualSquare --n_rollouts 20 "
        "--dataset_path dataset/dual_square/rollouts.hdf5 --dataset_obs\n"
        "Then extract red-nut snapshots → dataset/dual_square/red_nut_on_peg.hdf5\n"
        "and gold-nut snapshots → dataset/dual_square/gold_nut_on_peg.hdf5\n"
        "using helper_scripts/create_guidance_dataset.py"
    ),
    "dual_gold": (
        "See dual_red hint — same data files, roles are just swapped in the config."
    ),
}


def _apply_ood_mode(args):
    """
    Resolve --ood_mode into concrete --env and --exp_setup_config values.
    Explicit user flags always take priority over the auto-resolved defaults.
    """
    if args.ood_mode == "none":
        return

    # Auto-select environment unless the user already overrode it
    if args.env is None:
        args.env = _OOD_ENV_MAP[args.ood_mode]
        print(f"[OOD] Auto-selected environment: {args.env}")

    # Auto-select config unless the user already provided one
    if args.exp_setup_config is None:
        default_cfg = _OOD_CONFIG_MAP[args.ood_mode]
        if default_cfg and os.path.exists(default_cfg):
            args.exp_setup_config = default_cfg
            print(f"[OOD] Using default config: {args.exp_setup_config}")
        else:
            print(
                f"[OOD] WARNING: No exp_setup_config found at '{default_cfg}'.\n"
                f"       You need to create guidance data first.\n"
                f"       {_OOD_COLLECT_HINT.get(args.ood_mode, '')}"
            )

    # Append ood_mode tag to output folder so results are kept separate
    if args.output_folder is not None:
        tag = args.ood_mode
        if not args.output_folder.endswith(tag):
            args.output_folder = os.path.join(args.output_folder, tag)
        print(f"[OOD] Output folder set to: {args.output_folder}")


def run_trained_agent(args):
    global MAIN_CAMERA

    # --- OOD experiment wiring (must run before anything else reads args.env / args.exp_setup_config)
    _apply_ood_mode(args)

    MAIN_CAMERA = args.camera_names[0] if args.camera_names is not None else "agentview_image"
    states_list, actions_list, labels_list = list(), list(), list()
    script_start_time = time.time()

    if args.output_folder is not None and not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    # Automatically handle output path defaults and explicitly disabled flags via "false"
    video_disabled = (args.video_path is not None and args.video_path.lower() == "false")
    dataset_disabled = (args.dataset_path is not None and args.dataset_path.lower() == "false")
    json_disabled = (args.json_path is not None and args.json_path.lower() == "false")

    if video_disabled: args.video_path = None
    if dataset_disabled: args.dataset_path = None
    if json_disabled: args.json_path = None

    if args.output_folder:
        env_suffix = args.env.lower() if args.env else "env"
        if args.video_path is None and not video_disabled:
            args.video_path = os.path.join(args.output_folder, f"dynaguide_{env_suffix}_vid.mp4")
        if args.dataset_path is None and not dataset_disabled:
            args.dataset_path = os.path.join(args.output_folder, f"dynaguide_{env_suffix}_rollouts.hdf5")
        if args.json_path is None and not json_disabled:
            args.json_path = os.path.join(args.output_folder, f"dynaguide_{env_suffix}_stats.json")

    # check if video writing is requested
    write_video = (args.video_path is not None)
    
    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.dp_model)
    
    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    
    # Load configuration object early to extract parameter metadata
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    exp_setup_config = None 
    if args.exp_setup_config is not None: 
        with open(args.exp_setup_config, "r") as f:
            exp_setup_config = json.load(f)
            
    good_dataset_path = args.good_states if exp_setup_config is None else exp_setup_config["pos_examples"]

    ############# classifier guidance ################
    # Extract structural dimensions dynamically from robomimic metadata
    shape_meta = ckpt_dict.get("shape_metadata", {})
    
    ACTION_DIM = shape_meta.get("action", {}).get("shape", [7])[0]
    
    # Try common Diffusion Policy config paths safely without triggering robomimic's lock error
    ACTION_CHUNK_LENGTH = 16  # Safe fallback baseline
    try:
        ACTION_CHUNK_LENGTH = config.algo.pred_horizon
    except (RuntimeError, AttributeError):
        try:
            ACTION_CHUNK_LENGTH = config.algo.horizon.prediction_horizon
        except (RuntimeError, AttributeError):
            pass
        
    # Read activated RGB camera keys safely from configuration modalities
    try:
        if config.observation.modalities.obs.rgb:
            cameras = config.observation.modalities.obs.rgb
        else:
            cameras = [MAIN_CAMERA]
    except (RuntimeError, AttributeError):
        cameras = [MAIN_CAMERA]

    # Filter cameras against what is actually stored in the dataset to prevent KeyErrors
    if good_dataset_path and os.path.exists(good_dataset_path):
        try:
            with h5py.File(good_dataset_path, "r") as f:
                first_demo = list(f["data"].keys())[0]
                available_obs_keys = list(f["data"][first_demo]["obs"].keys())
                cameras = [cam for cam in cameras if cam in available_obs_keys]
        except Exception:
            pass

    # Safe fallback if intersection filtering leaves the list empty
    if not cameras:
        cameras = [MAIN_CAMERA]
        
    # Handle low-dim proprioception parsing
    all_shapes = shape_meta.get("all_shapes", {})
    proprio_key = next((k for k in all_shapes.keys() if "proprio" in k.lower()), None)
    
    if proprio_key is not None:
        proprio = "proprio"
        proprio_dim = all_shapes[proprio_key][0]
    elif all(k in all_shapes for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]):
        proprio = "proprio"
        proprio_dim = all_shapes["robot0_eef_pos"][0] + all_shapes["robot0_eef_quat"][0] + all_shapes["robot0_gripper_qpos"][0]
    else:
        proprio = None
        proprio_dim = 9  # Fallback workspace baseline dimension
        
    padding = True
    pad_mode = "repeat"

    model = FinalStatePredictionDino(ACTION_DIM, ACTION_CHUNK_LENGTH, cameras=cameras, reconstruction = True, \
                                     proprio = proprio, proprio_dim = proprio_dim)

    _ckpt = torch.load(args.dynamics_model)
    if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt:
        _ckpt = _ckpt["model_state_dict"]
    model.load_state_dict(_ckpt)
    model.to("cuda")
    model.eval()

    exp_setup_config = None 
    if args.exp_setup_config is not None: 
        with open(args.exp_setup_config, "r") as f:
            exp_setup_config = json.load(f)
    
    # ensure backward compatibility 
    good_dataset_path = args.good_states if args.exp_setup_config is None else exp_setup_config["pos_examples"]
    good_dataset = None 
    if exp_setup_config["pos_examples"] is not None:
        good_dataset = MultiviewDataset(good_dataset_path, action_chunk_length = ACTION_CHUNK_LENGTH, cameras = cameras, \
                                        padding = padding, pad_mode = pad_mode, proprio = proprio)
    
    bad_dataset = None 
    if exp_setup_config["use_neg"]:
        bad_dataset_path = exp_setup_config["neg_examples"]

        bad_dataset = MultiviewDataset(bad_dataset_path, action_chunk_length = ACTION_CHUNK_LENGTH, cameras = cameras, \
                                        padding = padding, pad_mode = pad_mode, proprio = proprio)
    
    if args.direct_position_guidance:
        classifier_grad, good_embeddings, bad_embeddings = calculate_position_guidance(exp_setup_config["loc_target"], args.scale)
    
    else:
        classifier_grad, good_embeddings, bad_embeddings = calculate_classifier_guidance(model, good_dataset = good_dataset, main_camera = MAIN_CAMERA, \
                                                        scale = args.scale, bad_dataset = bad_dataset, alpha = args.alpha, max_examples = args.max_examples)
    # Wrap the guidance function to handle both bundled and split robomimic low-dim keys
    base_guidance_func = classifier_grad
    
    def guidance_wrapper(states, actions):
        if "proprio" not in states:
            # 1. Look for any bundled key containing 'proprio' case-insensitively
            proprio_key = next((k for k in states.keys() if "proprio" in k.lower()), None)
            if proprio_key is not None:
                states["proprio"] = states[proprio_key]
            
            # 2. If split up into standard robomimic low-dim keys, concatenate them into a 9D vector
            elif all(k in states for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]):
                states["proprio"] = torch.cat([
                    states["robot0_eef_pos"],
                    states["robot0_eef_quat"],
                    states["robot0_gripper_qpos"]
                ], dim=-1)
            else:
                raise KeyError(
                    f"Could not find or reconstruct 'proprio' key in states. "
                    f"Available keys are: {list(states.keys())}"
                )
        # Enable non-deprecated Mixed Precision Autocast
        with torch.amp.autocast('cuda', enabled=True):
            return base_guidance_func(states, actions)
    
    classifier_grad = guidance_wrapper

    robot_initial_positions = None  
    if exp_setup_config["reset_poses"] is not None: 
        with open(exp_setup_config["reset_poses"], "r") as f: # for button and door right 
            poses = json.load(f) 
            robot_initial_positions = [np.array(k) for k in poses["robot_states"]]

    with open(args.output_folder + "/args.json", "w") as f:
        json.dump(vars(args), f) # tracks everything that runs this program 

    # read rollout settings
    rollout_horizon = args.horizon
    if args.n_rollouts is None:
        if robot_initial_positions is not None:
            rollout_num_episodes = len(robot_initial_positions)
            use_sequential_poses = True
        else:
            rollout_num_episodes = 20
            use_sequential_poses = False
    else:
        rollout_num_episodes = args.n_rollouts
        use_sequential_poses = True
    
    if "proprio" not in config.observation.modalities.obs.low_dim:
        config.observation.modalities.obs.low_dim.append("proprio")

    if rollout_horizon is None:
        # read horizon from config
        rollout_horizon = config.experiment.rollout.horizon


    # Tight-fit experiment: inject hole dimensions into env metadata before construction.
    if args.hole_x is not None or args.hole_y is not None:
        if args.env is None:
            args.env = "NutAssemblyTightSquare"
        hx = args.hole_x if args.hole_x is not None else 0.018
        hy = args.hole_y if args.hole_y is not None else 0.018
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_x"] = hx
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_y"] = hy

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=(args.video_path is not None),
        verbose=True,
    )

    universal_state = env.get_state() if args.same_env else None

    # Auto-fill camera rendering info if not specified
    if args.camera_names is None:
        # We fill in the automatic values
        env_type = EnvUtils.get_env_type(env=env)
        args.camera_names = DEFAULT_CAMERAS[env_type]
  

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_dir = os.path.dirname(args.video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        video_writer = imageio.get_writer(args.video_path, fps=20)
    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        # Create parent directories if they don't exist
        dataset_dir = os.path.dirname(args.dataset_path)
        if dataset_dir:
            os.makedirs(dataset_dir, exist_ok=True)
            
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        # If we already computed good/bad embeddings for guidance, save them at the root
        try:
            if 'good_embeddings' in locals() and good_embeddings is not None:
                ge = good_embeddings.detach().cpu().numpy()
                data_writer.create_dataset('good_embeddings', data=ge, compression='gzip')
            if 'bad_embeddings' in locals() and bad_embeddings is not None:
                be = bad_embeddings.detach().cpu().numpy()
                data_writer.create_dataset('bad_embeddings', data=be, compression='gzip')
        except Exception as _e:
            print(f"Warning: could not write global embeddings to HDF5: {_e}")
        total_samples = 0

    rollout_stats = []
    for i in tqdm(range(rollout_num_episodes)):
        try:
            stats, traj = rollout(
                policy=policy, 
                env=env, 
                horizon=rollout_horizon, 
                video_writer=video_writer, 
                video_skip=args.video_skip, 
                return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                classifier_grad = classifier_grad,
                reset_state = universal_state,
                model = model,
                good_embeddings = good_embeddings,
                bad_embeddings = bad_embeddings,
                exp_dir = args.output_folder,
                rollout_count = i,
                setup_config = exp_setup_config,
                save_frames = args.save_frames,
                goal_dir = args.goal_dir,
                ss = args.ss,
                initial_positions = robot_initial_positions,
                render_denoising = args.render_denoising,
                use_sequential_poses = use_sequential_poses,
                pose_index = i,
                save_images = args.save_images
            )
        except KeyboardInterrupt:
            sys.exit(0)
        
        rollout_stats.append(stats)
        print(f"[Timing] Rollout {i + 1} / {rollout_num_episodes} finished in: {stats['time']:.2f} seconds")

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            # If observations were recorded, compute and save latent embeddings (current and predicted)
            if args.dataset_obs:
                try:
                    # determine main camera key used for rendering/observations
                    main_cam = args.camera_names[0] if args.camera_names is not None else MAIN_CAMERA
                    imgs = np.array(traj["obs"][main_cam])  # T x [stack] x H x W x C or T x H x W x C (uint8)
                    actions_arr = np.array(traj["actions"])  # T x action_dim

                    # optional proprio: try common keys and reconstruct if necessary
                    proprio_arr = None
                    try:
                        if "proprio" in traj["obs"]:
                            proprio_arr = np.array(traj["obs"]["proprio"])  # T x [stack] x proprio_dim or T x proprio_dim
                        else:
                            # look for any key containing 'proprio'
                            obs_keys = list(traj["obs"].keys())
                            pkey = next((k for k in obs_keys if 'proprio' in k.lower()), None)
                            if pkey is not None:
                                proprio_arr = np.array(traj["obs"][pkey])
                            else:
                                # try reconstructing from common robomimic low-dim keys
                                parts = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
                                if all(p in traj["obs"] for p in parts):
                                    parts_arrs = [np.array(traj["obs"][p]) for p in parts]
                                    # collapse frame stacking for each part if present
                                    parts_arrs = [a[:, -1] if a.ndim == 3 else a for a in parts_arrs]
                                    proprio_arr = np.concatenate(parts_arrs, axis=-1)
                    except Exception:
                        proprio_arr = None

                    # collapse frame stacking to latest observation if present
                    if imgs.ndim == 5:
                        imgs = imgs[:, -1]
                    if proprio_arr is not None and proprio_arr.ndim == 3:
                        proprio_arr = proprio_arr[:, -1]

                    seq_len = ACTION_CHUNK_LENGTH
                    device = next(model.parameters()).device
                    curr_list = []
                    pred_list = []
                    T = imgs.shape[0]
                    if T >= seq_len:
                        for t_idx in range(0, T - seq_len + 1):
                            img_t = imgs[t_idx]
                            action_chunk = actions_arr[t_idx : t_idx + seq_len]

                            # prepare tensors
                            img_tensor = torch.tensor(img_t).permute(2, 0, 1).unsqueeze(0).float().to(device)
                            state = {main_cam: img_tensor}
                            if proprio_arr is not None:
                                state["proprio"] = torch.tensor(proprio_arr[t_idx]).unsqueeze(0).float().to(device)

                            actions_tensor = torch.tensor(action_chunk).unsqueeze(0).float().to(device)

                            with torch.no_grad():
                                curr_e = model.state_embedding(state).flatten(start_dim=1).cpu().numpy()
                                pred_e = model.state_action_embedding(state, actions_tensor).flatten(start_dim=1).cpu().numpy()

                            curr_list.append(curr_e[0])
                            pred_list.append(pred_e[0])

                    if len(curr_list) > 0:
                        emb_grp = ep_data_grp.create_group("embeddings")
                        emb_grp.create_dataset("current", data=np.stack(curr_list).astype(np.float32), compression="gzip")
                        emb_grp.create_dataset("predicted", data=np.stack(pred_list).astype(np.float32), compression="gzip")
                except Exception as e:
                    print(f"Warning: failed to compute/save embeddings for demo_{i}: {e}")
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    # Pop non-numeric per-nut/yaw fields before numeric aggregation
    nut_assembly_log = []
    yaw_log = []
    for s in rollout_stats:
        entry = {}
        if 'assembly_yaw_deg' in s:
            entry['assembly_yaw_deg'] = s.pop('assembly_yaw_deg')
            yaw_log.append(entry['assembly_yaw_deg'])
        if 'assembled_nut' in s:
            entry['assembled_nut'] = s.pop('assembled_nut')
        if entry:
            nut_assembly_log.append(entry)

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Time_Episode"] = np.sum(rollout_stats["time"]) / 60.
    avg_rollout_stats["Num_Episode"] = len(rollout_stats["Success_Rate"])

    # Insertion angle and orientation cluster summary (all NutAssembly environments)
    valid_yaws = [y for y in yaw_log if y is not None]
    if valid_yaws:
        from collections import Counter
        clusters = [get_orientation_cluster(y) for y in valid_yaws]
        cluster_counts = Counter(clusters)
        total_success = len(valid_yaws)
        print("\n=== Insertion Angle Summary ===")
        print(f"  All yaw angles (deg): {[round(y, 1) for y in valid_yaws]}")
        print(f"  Mean: {np.mean(valid_yaws):.1f}°  Std: {np.std(valid_yaws):.1f}°  "
              f"Min: {np.min(valid_yaws):.1f}°  Max: {np.max(valid_yaws):.1f}°")
        print("\n  Orientation Clusters:")
        for cluster in CLUSTER_ORDER:
            count = cluster_counts.get(cluster, 0)
            bar = "#" * count
            print(f"    {cluster:<14}: {count:3d}/{total_success} ({100*count/total_success:.0f}%)  {bar}")
        avg_rollout_stats["All_Insertion_Yaws_Deg"] = valid_yaws
        avg_rollout_stats["Insertion_Yaw_Mean_Deg"] = float(np.mean(valid_yaws))
        avg_rollout_stats["Insertion_Yaw_Std_Deg"] = float(np.std(valid_yaws))
        avg_rollout_stats["Orientation_Cluster_Counts"] = dict(cluster_counts)

    # Per-nut summary (NutAssemblyDualSquare only)
    dual_log = [e for e in nut_assembly_log if 'assembled_nut' in e]
    if dual_log:
        from collections import Counter
        assembled = [e['assembled_nut'] for e in dual_log]
        counts = Counter(str(a) for a in assembled)
        total = len(dual_log)
        print("\n=== Per-Nut Assembly Summary ===")
        # CLUSTER_ORDER defined at module level
        for label, count in sorted(counts.items()):
            yaws = [e['assembly_yaw_deg'] for e in dual_log
                    if str(e['assembled_nut']) == label and e.get('assembly_yaw_deg') is not None]
            display = "No assembly" if label == "None" else label
            yaw_str = f"  mean yaw={np.mean(yaws):.1f}° (std {np.std(yaws):.1f}°)" if yaws else ""
            print(f"  {display}: {count}/{total} rollouts ({100*count/total:.0f}%){yaw_str}")
            if yaws:
                nut_clusters = Counter(get_orientation_cluster(y) for y in yaws)
                for cl in CLUSTER_ORDER:
                    c = nut_clusters.get(cl, 0)
                    bar = "#" * c
                    print(f"      {cl:<22}: {c:3d}/{len(yaws)} ({100*c/len(yaws):.0f}%)  {bar}")
        avg_rollout_stats["Nut_Counts"] = dict(counts)
        avg_rollout_stats["Assembly_Yaw_Mean_Deg"] = {
            label: float(np.mean([e['assembly_yaw_deg'] for e in dual_log
                                   if str(e['assembled_nut']) == label and e.get('assembly_yaw_deg') is not None]))
            for label in counts if label != "None"
        }

    print("Average Rollout Stats")
    stats_json = json.dumps(avg_rollout_stats, indent=4)
    print(stats_json)
    if args.json_path is not None:
        json_f = open(args.json_path, "w")
        json_f.write(stats_json)
        json_f.close()

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))

    print(f"[Timing] Full run-through completed in: {time.time() - script_start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--dp_model",
        type=str,
        default = None,
        help="path to saved diffusion policy checkpoint pth file",
    )

    # Path to dynamics model
    parser.add_argument(
        "--dynamics_model",
        type=str,
        default = None,
        help="path to dynamics model checkpoint",
    )

    parser.add_argument(
        "--good_states",
        type=str,
        default = None,
        help="examples of good states",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=None,
        help="Number of rollouts to run. If omitted, runs all poses sequentially.",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--save_frames",
        action='store_true',
        help="you save every frame of the guidance visual",
    )

    parser.add_argument(
        "--direct_position_guidance",
        action='store_true',
        help="use position guidance (https://arxiv.org/abs/2411.16627)",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="set 'false' to disable video rendering (optional). Uses output_folder as fallback (recommended)",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="set 'false' to disable hdf5 file cration of the rollouts (optional). Uses output_folder as fallback (recommended)",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default, observations are excluded and only simulator states are saved)",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--same_env",
        action='store_true',
        help="reset to the same environment every time",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="How much to influence",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--alpha",
        type=float,
        default=30,
        help="softness of guidance consideration",
    )
    
    # for seeding before starting rollouts
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="softness of guidance consideration",
    )

    # Dump a json of the rollout results stats to the specified path
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="set 'false' to disable saving results.json (optional). Uses output_folder as fallback (recommended)",
    )

    parser.add_argument(
        "--exp_setup_config",
        type=str,
        default=None,
        help="(optional) dump a json of the rollout results stats to the specified path",
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Reccomended: Define a folder to save all created files & results",
    )

    # where the goals are stored; not needed if you're not training goal conditioned models 
    parser.add_argument(
        "--goal_dir",
        type=str,
        default=None,
        help="(optional) if provided, override the output folder path defined in the config",
    )

    # Dump a file with the error traceback at this path. Only created if run fails with an error.
    parser.add_argument(
        "--error_path",
        type=str,
        default=None,
        help="(optional) dump a file with the error traceback at this path. Only created if run fails with an error.",
    )

    parser.add_argument(
        "--ss",
        type=int,
        default=4,
        help="Stochastic Sampling guidance steps per environment interaction step",
    )
    parser.add_argument(
        "--render_denoising",
        action='store_true',
        help="Render and include the denoising collage grid in the output video",
    )
    parser.add_argument(
        "--save_images",
        action='store_true',
        help="Explicitly preserve and save raw image observations to the HDF5 rollout file database",
    )

    # -----------------------------------------------------------------------
    # Tight-fit experiment
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--hole_x", type=float, default=None,
        help="inner hole half-extent in x (metres) for NutAssemblyTightSquare; "
             "omit to use the standard nut geometry",
    )
    parser.add_argument(
        "--hole_y", type=float, default=None,
        help="inner hole half-extent in y (metres) for NutAssemblyTightSquare; "
             "omit to use the standard nut geometry",
    )

    # -----------------------------------------------------------------------
    # OOD experiment selection
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--ood_mode",
        type=str,
        default="none",
        choices=["none", "red_nut", "dual_red", "dual_gold"],
        help=(
            "Select an OOD evaluation experiment. "
            "Automatically sets --env and (if present on disk) --exp_setup_config:\n"
            "  none       — standard NutAssemblySquare (default, no change)\n"
            "  red_nut    — NutAssemblyRedSquare: same geometry, red color; tests visual OOD robustness\n"
            "  dual_red   — NutAssemblyDualSquare: two nuts on table, steer toward the RED nut\n"
            "  dual_gold  — NutAssemblyDualSquare: two nuts on table, steer toward the GOLD nut\n"
            "Explicit --env or --exp_setup_config flags always override these defaults."
        ),
    )

    args = parser.parse_args()
    res_str = None
    try:
        run_trained_agent(args)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        if args.error_path is not None:
            f = open(args.error_path, "w")
            f.write(res_str)
            f.close()
        raise e
