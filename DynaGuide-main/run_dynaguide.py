import argparse
import os
import json
import h5py
import imageio
import sys
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import math
import re
from datetime import datetime

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
from core.calvin_utils import generate_reset_state
from core.calvin_behaviors import make_tracker, validate_experiment_config, summarize_behaviors
from core.dynaguide import calculate_classifier_guidance, calculate_position_guidance

"""
THIS CODE ADAPTED FROM ROBOMIMIC POLICY ROLLOUT CODE
"""


MAIN_CAMERA = "third_person"

# what a run writes into its output folder, which an earlier run must not have there already
OUTPUT_FILES = re.compile(r"args\.json|\d+_\d+\.png|.*\.(mp4|hdf5)|.*_log\.json")


def prepare_output_folder(args):
    """Everything of a run goes into output_folder, named by date and time. Stops instead of overwriting an earlier run."""
    os.makedirs(args.output_folder, exist_ok=True)
    existing = [os.path.join(args.output_folder, f) for f in sorted(os.listdir(args.output_folder)) if OUTPUT_FILES.fullmatch(f)]
    existing += [path for path in [args.video_path, args.dataset_path, args.json_path] if path is not None and os.path.exists(path)]
    if len(existing) > 0:
        sys.exit(f"Output folder {args.output_folder} already has results, choose a new folder so they aren't overwritten: {existing}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.video_path = args.video_path or os.path.join(args.output_folder, f"{timestamp}_video.mp4")
    args.dataset_path = args.dataset_path or os.path.join(args.output_folder, f"{timestamp}_rollout.hdf5")
    args.json_path = args.json_path or os.path.join(args.output_folder, f"{timestamp}_log.json")
    args.dataset_obs = True # the success evaluation needs them
    for path in [args.video_path, args.dataset_path, args.json_path]:
        if os.path.dirname(path) != "":
            os.makedirs(os.path.dirname(path), exist_ok=True)


def render_video_frame(env, camera_names, collage):
    video_img = []
    for cam_name in camera_names:
        img = env.render(mode="rgb_array", height=128 * 5, width=128 * 5, camera_name=cam_name)
        video_img.append(img)
    video_img = np.concatenate(video_img, axis=0) # concatenate horizontally
    video_img = cv2.resize(video_img, (128 * 5, 128 * 5))
    if collage is not None:
        video_img = np.concatenate((collage, video_img), axis = 0)
    return video_img

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
    for i in range(0, len(actions_list), SKIP_RATE):
        action = actions_list[i]
        with torch.no_grad():
            s_end_embedding, reco_image = model(fed_state, action) # gets the s, a embedding only 
            reco_image = torch.clip(resizer(reco_image), 0, 1)
        
        s_end_embedding = s_end_embedding.flatten(start_dim = 1)
        if target_embeddings is not None:
            s_norm = torch.cdist(target_embeddings, s_end_embedding, p=2.0)
            sa_average_norm = -torch.logsumexp(-s_norm / alpha, dim = 0).detach().cpu().numpy()

        reco_image = np.transpose(np.clip(reco_image[0].detach().cpu().numpy(), 0, 1), (1, 2, 0)).copy() * 255
        reco_image = reco_image.astype(np.uint8)
        if target_embeddings is not None:
            reco_image = cv2.putText(reco_image, str(int(sa_average_norm.item())), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        predicted_image_list.append(reco_image.copy()) # copy is needed for the tiling 

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
            setup_config = None, save_frames = False, goal_dir = None, ss = 4, alpha = 20, initial_positions = None, no_collage = False, keep_images = False):
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
    - no_collage: leave the dynamics model reconstructions out of the video, which skips their forward passes
    - keep_images: also return the images in traj["obs"], so create_guidance_conditions.py can make conditions from this rollout
    """
    rollout_timestamp = time.time()
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)

    policy.start_episode()

    obs = env.reset()   

    # for goal conditioning baseline 
    if goal_dir is not None:
        env.set_goal(im_paths = goal_dir)
   
    # this is for calvin envrionment     
    special_state, articulated_binaries = generate_reset_state(sim_hold = setup_config["env_setup"]) # and keep the switch off 

    if initial_positions is not None:
        state_to_reset = {"scene" : special_state, "robot" : random.choice(initial_positions)}
    else:
        state_to_reset = special_state 

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
    time_policy, time_env, time_collage, time_video = 0., 0., 0., 0. # seconds per part of the rollout, for the results json
    collage = None
    goal_tracker = make_tracker(setup_config, articulated_binaries) # decides when the rollout ends, depending on the experiment config
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):
            t1 = time.time()
            if goal_dir is not None: # goal conditoning 
                goal_dict = env.get_goal()
                act = policy(ob=obs, guidance_function = classifier_grad, goal = goal_dict, guidance_type = "diffusion", ss = ss)
            else:
                act = policy(ob=obs, guidance_function = classifier_grad, guidance_type = "diffusion", ss = ss) # this is where the guidance happens 

            t2 = time.time()
            time_policy += t2 - t1

            next_obs, r, done, _ = env.step(act)
            time_env += time.time() - t2

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            if step_i % 14 == 0: # after every resampling, update the visuals 
                denoising_list = policy.policy.corrections_list 
                guidance_list = policy.policy.guidance_list
                print("Guidance norms: ", [np.linalg.norm(guidance.detach().cpu().numpy()).item() for guidance in guidance_list])
                if not no_collage:
                    t3 = time.time()
                    target_embeddings = bad_embeddings if good_embeddings is None else good_embeddings
                    collage = visualize_guidance_process(model, denoising_list, obs, target_embeddings, alpha = alpha)
                    time_collage += time.time() - t3
         
            if video_writer is not None:
                if video_count % video_skip == 0:
                    t3 = time.time()
                    video_img = render_video_frame(env, camera_names, collage)
                    video_writer.append_data(video_img)
                    time_video += time.time() - t3
                if step_i % 14 == 0 and save_frames:
                    print("SAVING!!")
                    plt.imsave(exp_dir + f"/{rollout_count}_{step_i}.png", video_img.astype(np.uint8))
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            # traj["states"].append(state_dict["states"])
            traj["states"].append(0) # HACK: don't assume access to states
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                if not keep_images:
                    del obs["third_person"] # don't keep the images for sake of space 
                    del obs["eye_in_hand"]
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))
         
            # FOR CALVIN ENVIRONMENT
            fed_state = obs["states"]
            fed_proprio = obs["proprio"]
            if len(obs["states"].shape) == 2: # collapse this if there's framestacking for the policy 
                fed_state = fed_state[-1] #obs["states"][-1]
                fed_proprio = fed_proprio[-1]

            done = goal_tracker.update(fed_state, fed_proprio)

            # break if done or if success
            if done or success:
                # always capture the last frame
                video_writer.append_data(render_video_frame(env, camera_names, collage))
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
        Time_Policy=time_policy,
        Time_Env=time_env,
        Time_Collage=time_collage,
        Time_Video=time_video,
    )

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
                  # HACK for removing frame stacking 
                if (k == "obs" or k == "next_obs") and len(traj[k]["states"][0].shape) > 1:
                    traj[k][kp] = traj[k][kp][:, -1] # 
        else:
            traj[k] = np.array(traj[k])

    return stats, traj, goal_tracker.behaviors


def run_trained_agent(args):
    states_list, actions_list, labels_list = list(), list(), list()

    prepare_output_folder(args)

    # some arg checking
    write_video = (args.video_path is not None)

    
    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)
    
    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    ############# classifier guidance ################
  # this needs to be aligned with the action chunk length in the trained model 
    ACTION_DIM = 7 
    ACTION_CHUNK_LENGTH = 16 # this is how long the action predictions are
    proprio_dim = 15 
    proprio = "proprio" # set to None if you want to exclude propriorception 
    cameras = [MAIN_CAMERA] # you can change this; it's hardcoded
    padding = True
    pad_mode = "repeat" #"zeros" #"repeat" # "zeros" for calvin 

    model = FinalStatePredictionDino(ACTION_DIM, ACTION_CHUNK_LENGTH, cameras=cameras, reconstruction = True, \
                                     proprio = proprio, proprio_dim = proprio_dim)
    model.load_state_dict(torch.load(args.guidance))
    model.to("cuda")
    model.eval()

    exp_setup_config = None 
    if args.exp_setup_config is not None: 
        with open(args.exp_setup_config, "r") as f:
            exp_setup_config = json.load(f)
        validate_experiment_config(exp_setup_config) # catch a typo in the config before spending time on rollouts

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
        classifier_grad, good_embeddings, bad_embeddings = calculate_classifier_guidance(model, good_dataset = good_dataset, main_camera = MAIN_CAMERA, 
                                                        scale = args.scale, bad_dataset = bad_dataset, alpha = args.alpha, max_examples = args.max_examples)
    
    robot_initial_positions = None  
    if exp_setup_config["reset_poses"] is not None: 
        with open(exp_setup_config["reset_poses"], "r") as f: # for button and door right 
            poses = json.load(f) 
            robot_initial_positions = [np.array(k) for k in poses["robot_states"]]

    with open(args.output_folder + "/args.json", "w") as f:
        json.dump(vars(args), f) # tracks everything that runs this program 

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if rollout_horizon is None:
        # read horizon from config
        rollout_horizon = config.experiment.rollout.horizon


    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False, # args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    if args.no_live_render:
        env.env.unwrapped.disable_live_render()

    universal_state = env.get_state() if args.same_env else None

    # Auto-fill camera rendering info if not specified
    if args.camera_names is None:
        # We fill in the automatic values
        env_type = EnvUtils.get_env_type(env=env)
        args.camera_names = DEFAULT_CAMERAS[env_type]
  

    # maybe set seed
    if args.seed is not None:
        random.seed(args.seed) # picks the robot start pose and the random parts of the reset
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    episode_behaviors = []
    for i in tqdm(range(rollout_num_episodes)):
        try:
            stats, traj, behaviors = rollout(
                policy=policy, 
                env=env, 
                horizon=rollout_horizon, 
                # render=args.render, 
                video_writer=video_writer, 
                video_skip=args.video_skip, 
                return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                # real=is_real_robot,
                # rate_measure=rate_measure,
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
                no_collage = args.no_collage,
                keep_images = args.dataset_images
            )
        except KeyboardInterrupt:
            sys.exit(0)
        
        rollout_stats.append(stats)
        episode_behaviors.append(behaviors)
        print(f"Rollout {i} behavior: {behaviors}")

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    is_image = k in ["third_person", "eye_in_hand"]
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip" if is_image else None)
                    if not is_image: # next_obs is only one frame ahead, so saving its images would double the file
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats.update(summarize_behaviors(episode_behaviors, exp_setup_config)) # replaces the environment's Success_Rate, which is always 0
    avg_rollout_stats["Time_Episode"] = np.sum(rollout_stats["time"]) / 60. # total time taken for rollouts in minutes
    avg_rollout_stats["Num_Episode"] = len(rollout_stats["Success_Rate"]) # number of episodes attempted
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default = None,
        help="path to saved checkpoint pth file",
    )

    # Path to trained model
    parser.add_argument(
        "--guidance",
        type=str,
        default = None,
        help="path to dynaguide checkpoint",
    )

        # Path to trained model
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
        default=27,
        help="number of rollouts",
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
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # # Whether to render rollouts to screen
    # parser.add_argument(
    #     "--render",
    #     action='store_true',
    #     help="on-screen rendering",
    # )

    parser.add_argument(
        "--dataset_images",
        action='store_true',
        help="also save the camera images of the rollouts in the dataset, so create_guidance_conditions.py can make guidance conditions from them. Makes the file much bigger",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--no_live_render",
        action='store_true',
        help="(optional) stop redrawing the PyBullet window to save time. The saved video and the images of the policy stay the same",
    )

    parser.add_argument(
        "--no_collage",
        action='store_true',
        help="(optional) leave the dynamics model reconstructions out of the video and the frames to save time",
    )

    parser.add_argument(
        "--save_frames",
        action='store_true',
        help="you save every frame of the guidance visual",
    )

    # Whether to render rollouts to screen
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
        help="(optional) render rollouts to this video file path",
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
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
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
        default=None,
        help="How much to influence",
    )

            # for seeding before starting rollouts
    parser.add_argument(
        "--alpha",
        type=float,
        default=20,
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
        help="(optional) dump a json of the rollout results stats to the specified path",
    )

    parser.add_argument(
        "--exp_setup_config",
        type=str,
        default=None,
        help="(optional) dump a json of the rollout results stats to the specified path",
    )
    
    # Dump a json of the rollout results stats to the specified path
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="folder for everything this run saves: video, rollouts, results json and frames. It must not hold an earlier run",
    )


    # where the goals are stored; not needed if you're not trianing goal conditioned models 
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

    # # If provided, do not run actions in env, and instead just measure the rate of action computation
    # parser.add_argument(
    #     "--hz",
    #     type=int,
    #     default=None,
    #     help="If provided, do not run actions in env, and instead just measure the rate of action computation and raise warnings if it dips below this threshold",
    # )

    parser.add_argument(
        "--ss",
        type=int,
        default=4,
        help="If provided, do not run actions in env, and instead just measure the rate of action computation and raise warnings if it dips below this threshold",
    )

    # # If provided, set num_inference_timesteps explicitly for diffusion policy evaluation
    # parser.add_argument(
    #     "--dp_eval_steps",
    #     type=int,
    #     default=None,
    #     help="If provided, set num_inference_timesteps explicitly for diffusion policy evaluation",
    # )

    args = parser.parse_args()
    res_str = None
    try:
        run_trained_agent(args)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        if args.error_path is not None:
            # write traceback to file
            f = open(args.error_path, "w")
            f.write(res_str)
            f.close()
        raise e
