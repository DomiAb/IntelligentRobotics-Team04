import argparse
import copy
import json
import os
import random
import shutil
import sys
from datetime import datetime

import cv2
import numpy as np

from calvin_env.envs.play_table_env import PlayTableSimEnv
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import BLOCK_MOVED, validate_experiment_config
from core.calvin_utils import BLOCK_INDEX, generate_reset_state

"""
Shows what the env_setup of an experiment config produces, without a policy or a rollout. It samples resets the same way
run_dynaguide.py does and renders them with the camera of the policy. Each reset is also simulated for a while without the
robot acting, to find layouts that don't hold: blocks that fall, shift, intersect or touch the robot.

Everything goes into a new folder <output_folder>/<config name>_<date and time>:
- resets_start.png: what the policy sees at the first step
- resets_settled.png: the same resets after settling. A red border means a check of that reset failed
- report.json: the start state, the checks and the movement of everything in each reset

Example, run from DynaGuide-main:
    python dynaguide_scripts/preview_resets.py --config calvin_exp_configs_examples/reset_layouts/blocks_in_all_places.json
The exit code is 1 if a check failed.
"""

REPO_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # DynaGuide-main, the scripts are in dynaguide_scripts
DEFAULT_ENV_CONFIG = os.path.join(REPO_FOLDER, "configs", "calvin.json")
DEFAULT_OUTPUT_FOLDER = os.path.join(REPO_FOLDER, "results", "reset_previews")

JOINTS = {"sliding_door" : 0, "drawer" : 1, "switch" : 3} # index in the lowdim state. They shouldn't move by themselves
JOINT_MOVED = 0.01
PENETRATION = 0.002 # blocks are allowed to touch, for example when they are stacked


def make_env(env_config_path, seed):
    """The environment of the experiments, with the camera images coming from the CPU so that this runs on any machine and without a window."""
    with open(env_config_path, "r") as f:
        env_kwargs = copy.deepcopy(json.load(f)["env_kwargs"])
    env_kwargs.update(show_gui = False, use_egl = False, seed = seed)
    return PlayTableSimEnv(**env_kwargs)


def load_robot_poses(config):
    if config.get("reset_poses") is None:
        return None
    with open(os.path.join(REPO_FOLDER, config["reset_poses"]), "r") as f:
        return [np.array(pose) for pose in json.load(f)["robot_states"]]


def bodies_in_contact(env, uid_a, uid_b, min_penetration = 0.0):
    contacts = env.p.getContactPoints(bodyA = uid_a, bodyB = uid_b, physicsClientId = env.cid)
    return any(contact[8] < -min_penetration for contact in contacts) if min_penetration > 0 else len(contacts) > 0


def check_contacts(env, block_uids):
    """Returns what is wrong in the current simulator state: blocks inside each other, or the robot touching a block."""
    problems = list()
    names = list(block_uids)
    for i, a in enumerate(names):
        if bodies_in_contact(env, block_uids[a], env.robot.robot_uid):
            problems.append(f"{a} touches the robot")
        for b in names[i + 1:]:
            if bodies_in_contact(env, block_uids[a], block_uids[b], min_penetration = PENETRATION):
                problems.append(f"{a} intersects {b}")
    return problems


def preview_one_reset(env, setup_config, robot_poses, settle_steps, render_size):
    """Resets like run_dynaguide.py does, and returns the two images and the report of this reset."""
    special_state, _ = generate_reset_state(sim_hold = setup_config["env_setup"])
    state_to_reset = special_state if robot_poses is None else {"scene" : special_state, "robot" : random.choice(robot_poses)}
    env.reset(state_to_reset)

    block_uids = {name : obj.uid for name, obj in zip(BLOCK_INDEX, env.scene.movable_objects)}
    start = env.get_state()
    problems = check_contacts(env, block_uids)
    image_start = env.render(height = render_size, width = render_size)[:, :, :3]

    for _ in range(settle_steps):
        env.p.stepSimulation(physicsClientId = env.cid)
        env.scene.step()
    settled = env.get_state()
    image_settled = env.render(height = render_size, width = render_size)[:, :, :3]

    moved = dict()
    for name, idx in BLOCK_INDEX.items():
        moved[name] = float(np.linalg.norm(settled[idx:idx + 3] - start[idx:idx + 3]))
        if moved[name] > BLOCK_MOVED:
            problems.append(f"{name} moved by {moved[name]:.3f} m")
    for name, idx in JOINTS.items():
        moved[name] = float(abs(settled[idx] - start[idx]))
        if moved[name] > JOINT_MOVED:
            problems.append(f"{name} moved by {moved[name]:.3f} m")
    problems = sorted(set(problems + check_contacts(env, block_uids)))

    report = {
        "start" : {
            **{name : float(start[idx]) for name, idx in JOINTS.items()},
            "green_light" : float(start[5]),
            **{name : [round(float(x), 4) for x in start[idx:idx + 3]] + [round(float(start[idx + 5]), 4)] for name, idx in BLOCK_INDEX.items()},
        },
        "moved_during_settling" : {name : round(value, 4) for name, value in moved.items()},
        "problems" : problems,
    }
    return image_start, image_settled, report


def label_tile(image, number, report):
    """Writes on the image which reset it is. A failed check gets a red border, and the first problem is named."""
    tile = image.astype(np.uint8).copy()
    door, drawer = report["start"]["sliding_door"], report["start"]["drawer"]
    text = f"#{number} door {'L' if door > 0.135 else 'R'} drawer {'open' if drawer > 0.08 else 'closed'}"
    cv2.putText(tile, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(tile, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if len(report["problems"]) > 0:
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (255, 0, 0), 5)
        problem = report["problems"][0] + (f" (+{len(report['problems']) - 1})" if len(report["problems"]) > 1 else "")
        cv2.putText(tile, problem, (6, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, problem, (6, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 80), 1, cv2.LINE_AA)
    return tile


def save_grid(path, tiles):
    columns = int(np.ceil(np.sqrt(len(tiles))))
    tiles = tiles + [np.zeros_like(tiles[0])] * (columns * int(np.ceil(len(tiles) / columns)) - len(tiles))
    rows = [np.concatenate(tiles[i:i + columns], axis = 1) for i in range(0, len(tiles), columns)]
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, axis = 0), cv2.COLOR_RGB2BGR))


def main(args):
    with open(args.config, "r") as f:
        setup_config = json.load(f)
    validate_experiment_config(setup_config) # stops at a typo before the simulator starts
    robot_poses = load_robot_poses(setup_config)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(args.output_folder, f"{config_name}_{timestamp}")
    os.makedirs(output_folder)
    shutil.copy(args.config, output_folder)

    env = make_env(args.env_config, args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tiles_start, tiles_settled, reports = list(), list(), list()
    for i in range(args.n):
        image_start, image_settled, report = preview_one_reset(env, setup_config, robot_poses, args.settle_steps, args.size)
        tiles_start.append(label_tile(image_start, i, report))
        tiles_settled.append(label_tile(image_settled, i, report))
        reports.append(report)
        if len(report["problems"]) > 0:
            print(f"Reset {i}: {report['problems']}")

    save_grid(os.path.join(output_folder, "resets_start.png"), tiles_start)
    save_grid(os.path.join(output_folder, "resets_settled.png"), tiles_settled)
    num_failed = sum(len(report["problems"]) > 0 for report in reports)
    with open(os.path.join(output_folder, "report.json"), "w") as f:
        json.dump({"config" : args.config, "seed" : args.seed, "num_resets" : args.n, "num_failed" : num_failed, "resets" : reports}, f, indent = 2)

    print(f"{args.n - num_failed} of {args.n} resets passed the checks. Saved to {output_folder}")
    return 1 if num_failed > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True, help = "experiment config with the env_setup to check")
    parser.add_argument("--output_folder", type = str, default = DEFAULT_OUTPUT_FOLDER, help = "the results get their own new folder inside this one")
    parser.add_argument("--n", type = int, default = 16, help = "number of resets")
    parser.add_argument("--seed", type = int, default = 0, help = "the same seed shows the same resets")
    parser.add_argument("--size", type = int, default = 300, help = "size of each image in pixels")
    parser.add_argument("--settle_steps", type = int, default = 240, help = "simulator steps to find unstable layouts, 240 is one second")
    parser.add_argument("--env_config", type = str, default = DEFAULT_ENV_CONFIG, help = "environment config")
    sys.exit(main(parser.parse_args()))
