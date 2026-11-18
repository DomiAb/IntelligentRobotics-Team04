import argparse
import json
import os
import random
import sys

import h5py
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_utils import BLOCK_INDEX, PLACES, STATE_WORDS, articulated_value, condition_files, generate_reset_state, parse_block
from core.guidance_conditions import Recording, Segment, _anchor_position, save_preview, write_conditions
from recombine_conditions import ANCHOR, DEFAULT_ENV_CONFIG, DRAWER_BLOCK, STATE_INDEX, Simulator, joint_word, make_env

"""
Renders robot poses into one fixed scene: the scene the experiment resets to, so the guidance conditions show exactly what the robot
will see, apart from the robot and what the goal changes. Made for tasks with one block, like calvin_exp_configs_examples/
multigoal/single_block_drawer.json, whose conditions come from calvin_exp_configs_examples/scene_conditions/single_block_drawer.json:

    python dynaguide_scripts/render_scene_conditions.py --spec calvin_exp_configs_examples/scene_conditions/single_block_drawer.json \
        --output_folder datasets/guidance_conditions/single_block_drawer

Run it from DynaGuide-main. The spec:

    "scene":  the env_setup of the experiment, with every joint and every block fixed (a spot like table_center, or "hidden")
    "block":  the block the task works with, like "blue"
    "sets":   {name : how to make one condition file}, each with
        "poses":      pose files (robot_poses or condition files, a path, a pattern or a list). The last frame of each demo is a pose
        "joints":     {joint : "pose" | a word | a number} where the condition differs from the scene. "pose" keeps the value of the
                      pose, like the drawer of a drawer_open pose. Joints that aren't listed are as in the scene
        "block":      where the block is: "start" (as in the scene), "pose" (where the block of the pose is, for poses of
                      extract_block_poses.py, which name it in the attribute "block"), or a place of calvin_utils.PLACES like
                      "drawer_center". A block in the drawer moves with the drawer
        "pose_block": for "block": "pose", which poses to take: "same" (only poses with the task's block, the default; the gripper is
                      as wide as that block) or "any" (any block, for poses where the block isn't in the gripper)
        "near_start": for "block": "pose", only poses whose block is this close (meters, in x and y) to the block's start position
        "max":        at most this many conditions, spread over the poses that work

A pose is skipped, with the reason in the manifest, if the robot would be inside the table, the block would be inside the robot or the
table (more than in the pose itself), or the gripper holds a joint that the condition moves.

The output folder: <set>.hdf5 and <set>.png for each set, and manifest.json with every condition, every skipped pose and why.
"""

SET_KEYS = {"poses", "joints", "block", "pose_block", "near_start", "max"}


def scene_state(scene):
    """The lowdim state of the scene. Raises an error unless every joint and block is fixed."""
    missing = [joint for joint in STATE_INDEX if joint not in scene]
    if len(missing) > 0:
        raise ValueError(f"scene: fix every joint, {missing} missing")
    blocks = scene.get("blocks", dict())
    for name in BLOCK_INDEX:
        if name not in blocks:
            raise ValueError(f"scene: fix every block, {name} missing (\"hidden\" takes it out of the scene)")
        low, high, _, yaw = parse_block(name, blocks[name])
        if not np.array_equal(low, high) or yaw is None:
            raise ValueError(f"scene: block {name} must be at a spot, not in a region, so every reset is the same")
    state, _ = generate_reset_state(scene, rng = random.Random(0))
    return state


def read_poses(spec):
    poses = list()
    for path in condition_files(spec):
        with h5py.File(path, "r") as file:
            for demo in sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1])):
                group = file["data"][demo]
                poses.append({"states" : group["obs/states"][-1].astype(np.float64), "proprio" : group["obs/proprio"][-1].astype(np.float64),
                              "action" : group["actions"][-1].astype(np.float64), "block" : group.attrs.get("block"),
                              "source" : str(group.attrs.get("source", f"{path}:{demo}")), "file" : f"{path}:{demo}"})
    return poses


def in_drawer(position):
    return DRAWER_BLOCK["x"][0] < position[0] < DRAWER_BLOCK["x"][1] and position[2] < DRAWER_BLOCK["z_below"]


def compose(scene, block, pose, entry):
    """The state of the condition, or (None, reason)."""
    state = scene.copy()
    for joint, value in entry.get("joints", dict()).items():
        state[STATE_INDEX[joint]] = pose["states"][STATE_INDEX[joint]] if value == "pose" else articulated_value(joint, value)
    state[2] = pose["states"][2] # the button springs back, keep it as the robot left it
    index = BLOCK_INDEX[block]
    where = entry.get("block", "start")
    if where == "pose":
        source = pose["block"]
        if source is None:
            return None, f"{pose['file']} names no block, take poses of extract_block_poses.py for \"block\": \"pose\""
        pose_index = BLOCK_INDEX[source]
        state[index : index + 6] = pose["states"][pose_index : pose_index + 6]
        if in_drawer(state[index : index + 3]): # the block lies in the drawer, it moves with it
            state[index + 1] -= state[STATE_INDEX["drawer"]] - pose["states"][STATE_INDEX["drawer"]]
    elif where != "start":
        low, _, inside, yaw = parse_block(block, where)
        state[index : index + 6] = np.concatenate([low, [0.0, 0.0, yaw if yaw is not None else 0.0]])
        if inside:
            state[index + 1] -= state[STATE_INDEX["drawer"]]
    return state, None


def held_joint_moved(pose, state, radius = 0.1):
    """A joint that the gripper is at and that the condition moves to the other side."""
    tcp = pose["proprio"][0:3]
    for joint, anchor in ANCHOR.items():
        near = np.linalg.norm(_anchor_position(anchor, pose["states"][None])[0] - tcp) < radius
        if near and joint_word(joint, state[STATE_INDEX[joint]]) != joint_word(joint, pose["states"][STATE_INDEX[joint]]):
            return joint
    return None


def render_set(sim, scene, block, name, entry, tolerance):
    unknown = set(entry) - SET_KEYS
    if len(unknown) > 0:
        raise ValueError(f"set {name}: unknown keys {sorted(unknown)}, choose from {sorted(SET_KEYS)}")
    start = scene[BLOCK_INDEX[block] : BLOCK_INDEX[block] + 2]
    images_states, kept, skipped = list(), list(), list()
    for pose in read_poses(entry["poses"]):
        note = {"source" : pose["source"], "file" : pose["file"]}
        if entry.get("block") == "pose":
            if entry.get("pose_block", "same") == "same" and pose["block"] != block:
                continue # a pose of another block, not a problem
            if "near_start" in entry:
                position = pose["states"][BLOCK_INDEX[pose["block"]] : BLOCK_INDEX[pose["block"]] + 2]
                if np.linalg.norm(position - start) > entry["near_start"]:
                    continue
        state, reason = compose(scene, block, pose, entry)
        if state is None:
            raise ValueError(f"set {name}: {reason}")
        joint = held_joint_moved(pose, state)
        if joint is not None:
            skipped.append({**note, "reason" : f"the gripper is at the {joint}, which this condition moves"})
            continue

        sim.set(pose["states"], pose["proprio"])
        before = sim.contacts()
        if entry.get("block") == "pose": # the block of the pose may have a different color, compare with it
            before = {(block if key[0] == pose["block"] else "_" + key[0], *key[1:]) : depth for key, depth in before.items()}
        sim.set(state, pose["proprio"])
        problems = [key for key, depth in sim.contacts().items() if key[0] in ["robot", block] and depth > before.get(key, 0.0) + tolerance]
        if len(problems) > 0:
            skipped.append({**note, "reason" : sim.describe(problems[0])})
            continue
        images_states.append((sim.render(), state.copy(), pose))
        kept.append({**note, "block" : [round(float(x), 4) for x in state[BLOCK_INDEX[block] : BLOCK_INDEX[block] + 3]],
                     "drawer" : round(float(state[STATE_INDEX["drawer"]]), 4), "robot_error" : round(sim.robot_error(pose["proprio"]), 5)})

    if "max" in entry and len(kept) > entry["max"]:
        chosen = np.linspace(0, len(kept) - 1, entry["max"]).round().astype(int)
        images_states, kept = [images_states[i] for i in chosen], [kept[i] for i in chosen]
    return images_states, kept, skipped


def main(args):
    if os.path.isdir(args.output_folder) and len(os.listdir(args.output_folder)) > 0:
        sys.exit(f"Output folder {args.output_folder} is not empty, choose a new folder so nothing gets overwritten")
    with open(args.spec, "r") as f:
        spec = json.load(f)
    block = spec["block"]
    if block not in BLOCK_INDEX:
        sys.exit(f"block must be one of {list(BLOCK_INDEX)}")
    scene = scene_state(spec["scene"])

    sim = Simulator(make_env(args.env_config))
    os.makedirs(args.output_folder, exist_ok = True)
    manifest = {"spec" : args.spec, "scene" : spec["scene"], "block" : block, "sets" : dict()}
    for name, entry in spec["sets"].items():
        images_states, kept, skipped = render_set(sim, scene, block, name, entry, args.tolerance)
        manifest["sets"][name] = {"conditions" : kept, "skipped" : skipped}
        print(f"{name}: {len(kept)} conditions, {len(skipped)} poses skipped")
        if len(kept) == 0:
            continue
        segments = list()
        for images, state, pose in images_states:
            recording = Recording(pose["source"], state[None], pose["proprio"][None], pose["action"][None], ["third_person", "eye_in_hand"],
                                  lambda camera, frames, images = images: np.stack([images[camera] for _ in frames]))
            segments.append(Segment(recording, 0, 0, name))
        write_conditions(os.path.join(args.output_folder, name + ".hdf5"), segments)
        save_preview(os.path.join(args.output_folder, name + ".png"), segments)
    with open(os.path.join(args.output_folder, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent = 4)
    print(f"Saved to {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type = str, required = True, help = "the scene, the block and the condition sets, see the top of this file")
    parser.add_argument("--output_folder", type = str, required = True, help = "new folder for the condition files")
    parser.add_argument("--tolerance", type = float, default = 0.002, help = "how much deeper than in the pose counts as inside, meters")
    parser.add_argument("--env_config", type = str, default = DEFAULT_ENV_CONFIG, help = "environment config")
    main(parser.parse_args())
