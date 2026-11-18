import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import BLOCK_LIFTED, COMPLETED
from core.calvin_utils import TABLE_Z
from core.guidance_conditions import (BLOCK_POSITION, BLOCK_REST_Z, GRIPPER_ACTION, GRIPPER_WIDTH, HELD_DISTANCE, HELD_WIDTH,
                                      Recording, Segment, read_calvin_folder, save_preview, write_conditions)
from recombine_conditions import DEFAULT_ENV_CONFIG, Simulator, make_env

"""
Collects robot poses that work with a block, like extract_robot_poses.py does for the drawer, door, switch and button. The poses go into
the robot_poses collection and are rendered again into any scene by render_scene_conditions.py.

    python dynaguide_scripts/extract_block_poses.py --sources datasets/calvin_debug_dataset/validation datasets/calvin_debug_dataset/training \
        datasets/guidance_conditions results --output_folder datasets/robot_poses/blocks

Run it from DynaGuide-main. A source is a split of a CALVIN dataset (episode_*.npz), an hdf5 file in the robomimic format (rollouts of
run_dynaguide.py, guidance conditions) or a folder, which is searched for hdf5 files. Images aren't needed, the states are enough.
Files that can't be read, like the rollout file of a run that was stopped, are skipped and listed in the manifest.

The kinds of poses, each frame of every recording is checked:
    block_lifted                 a block in the closed gripper, at least BLOCK_LIFTED (0.03) above the table top. What the robot
                                 shows right after it picked up a block from the table
    block_in_drawer_released     a block lying on the floor of the drawer, not in the gripper, the tcp above it within 0.12,
                                 and the drawer open at least as far as COMPLETED["drawer_open"]. What it shows right after it put a
                                 block into the drawer and let go

Which frames are kept:
- of each run of matching frames, the last frame and then every --min_gap frames further back, because neighbouring frames are the same
- a pose that is the same as one already kept is dropped: same kind, every arm joint within --duplicate_radians and the block within
  --duplicate_meters. The sources are read in the order given, so the debug set comes before the conditions that were cut from it

The output folder:
    <kind>.hdf5     one demo per pose, a single frame, rendered again in the simulator in its own scene. The demo attribute
                    "block" says which block the pose works with
    <kind>.png      the poses as rendered
    manifest.json   every pose with where it came from, and what was dropped and why
"""

KINDS = ["block_lifted", "block_in_drawer_released"]
RELEASED_TCP_RADIUS = 0.12
ARM_JOINTS = slice(7, 14) # in proprio


def read_states_hdf5(path):
    """The recordings of an hdf5 file as (name, states, proprio, actions), without images."""
    recordings = list()
    source = os.path.relpath(path)
    with h5py.File(path, "r") as file:
        for demo in sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1])):
            group = file["data"][demo]
            if "actions" not in group or "obs" not in group or "states" not in group["obs"] or "proprio" not in group["obs"]:
                continue
            recordings.append((f"{source}:{demo}", group["obs/states"][()].astype(np.float64), group["obs/proprio"][()].astype(np.float64),
                               group["actions"][()].astype(np.float64)))
    return recordings


def read_sources(sources):
    """Every recording of the sources, and the files that couldn't be read."""
    recordings, unreadable = list(), list()
    for source in sources:
        if os.path.isdir(source) and len(glob.glob(os.path.join(source, "episode_*.npz"))) > 0:
            recordings += [(f"{os.path.relpath(source)}:{r.name}", r.states, r.proprio, r.actions) for r in read_calvin_folder(source)]
            continue
        files = sorted(glob.glob(os.path.join(source, "**", "*.hdf5"), recursive = True)) if os.path.isdir(source) else [source]
        if len(files) == 0:
            sys.exit(f"{source} is neither a CALVIN split, nor an hdf5 file, nor a folder with hdf5 files")
        for path in files:
            try:
                recordings += read_states_hdf5(path)
            except (OSError, KeyError) as error:
                unreadable.append({"file" : path, "error" : str(error)[:200]})
    return recordings, unreadable


def frame_masks(states, proprio):
    """{(kind, color) : mask over the frames}."""
    masks = dict()
    tcp, width, gripper_action = proprio[:, 0:3], proprio[:, GRIPPER_WIDTH], proprio[:, GRIPPER_ACTION]
    for color, index in BLOCK_POSITION.items():
        position = states[:, index : index + 3]
        distance = np.linalg.norm(tcp - position, axis = 1)
        held = (distance < HELD_DISTANCE) & (width < HELD_WIDTH) & (gripper_action < 0)
        masks[("block_lifted", color)] = held & (position[:, 2] > TABLE_Z + BLOCK_LIFTED)
        on_drawer_floor = np.abs(position[:, 2] - BLOCK_REST_Z[0]) < 0.015
        masks[("block_in_drawer_released", color)] = on_drawer_floor & ~held & (distance < RELEASED_TCP_RADIUS) \
            & (tcp[:, 2] > position[:, 2]) & (states[:, 1] >= COMPLETED["drawer_open"])
    return masks


def candidate_frames(mask, min_gap):
    """The last frame of each run of matching frames, then every min_gap frames further back."""
    frames = np.flatnonzero(mask)
    if len(frames) == 0:
        return []
    picks = list()
    for run in np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1):
        picks += [int(run[position]) for position in range(len(run) - 1, -1, -min_gap)]
    return sorted(picks)


def find_duplicate(pose, kept, radians, meters):
    for other in kept:
        if np.abs(pose["proprio"][ARM_JOINTS] - other["proprio"][ARM_JOINTS]).max() < radians \
                and np.linalg.norm(pose["block_position"] - other["block_position"]) < meters:
            return other["info"]["source"]
    return None


def main(args):
    if os.path.isdir(args.output_folder) and len(os.listdir(args.output_folder)) > 0:
        sys.exit(f"Output folder {args.output_folder} is not empty, choose a new folder so nothing gets overwritten")
    recordings, unreadable = read_sources(args.sources)
    print(f"{len(recordings)} recordings, {len(unreadable)} files couldn't be read")

    kept = {kind : list() for kind in KINDS}
    dropped, found = list(), {kind : 0 for kind in KINDS}
    for name, states, proprio, actions in recordings:
        for (kind, color), mask in frame_masks(states, proprio).items():
            for frame in candidate_frames(mask, args.min_gap):
                index = BLOCK_POSITION[color]
                pose = {"states" : states[frame], "proprio" : proprio[frame], "action" : actions[frame], "block" : color,
                        "block_position" : states[frame, index : index + 3],
                        "info" : {"source" : f"{name}:{frame}", "block" : color, "drawer" : round(float(states[frame, 1]), 4),
                                  "block_position" : [round(float(x), 4) for x in states[frame, index : index + 3]],
                                  "tcp" : [round(float(x), 4) for x in proprio[frame, 0:3]]}}
                found[kind] += 1
                same = find_duplicate(pose, kept[kind], args.duplicate_radians, args.duplicate_meters)
                if same is not None:
                    dropped.append({**pose["info"], "kind" : kind, "reason" : f"same pose as {same}"})
                    continue
                kept[kind].append(pose)

    sim = Simulator(make_env(args.env_config))
    os.makedirs(args.output_folder, exist_ok = True)
    manifest = {"sources" : args.sources, "min_gap" : args.min_gap, "duplicate_radians" : args.duplicate_radians,
                "duplicate_meters" : args.duplicate_meters, "unreadable" : unreadable, "poses" : dict(), "dropped" : dropped}
    for kind in KINDS:
        if len(kept[kind]) == 0:
            print(f"{kind}: no poses found")
            continue
        segments = list()
        for pose in kept[kind]:
            sim.set(pose["states"], pose["proprio"])
            images = sim.render()
            recording = Recording(pose["info"]["source"], pose["states"][None], pose["proprio"][None], pose["action"][None],
                                  ["third_person", "eye_in_hand"],
                                  lambda camera, frames, images = images: np.stack([images[camera] for _ in frames]))
            segments.append(Segment(recording, 0, 0, kind))
        path = os.path.join(args.output_folder, kind + ".hdf5")
        write_conditions(path, segments)
        with h5py.File(path, "a") as file: # which block each pose works with, render_scene_conditions.py reads it
            for number, pose in enumerate(kept[kind]):
                file["data"][f"demo_{number}"].attrs["block"] = pose["block"]
        save_preview(os.path.join(args.output_folder, kind + ".png"), segments)
        manifest["poses"][kind] = [pose["info"] for pose in kept[kind]]
        colors = {color : sum(pose["block"] == color for pose in kept[kind]) for color in BLOCK_POSITION}
        print(f"{kind}: {len(kept[kind])} poses from {found[kind]} matching frames, by block {colors}")
    with open(os.path.join(args.output_folder, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent = 4)
    print(f"Saved to {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type = str, nargs = "+", required = True,
                        help = "CALVIN splits, hdf5 files, or folders that are searched for hdf5 files")
    parser.add_argument("--output_folder", type = str, required = True, help = "new folder for the pose files")
    parser.add_argument("--min_gap", type = int, default = 8, help = "frames between two candidates of one run of matching frames")
    parser.add_argument("--duplicate_radians", type = float, default = 0.05, help = "arm joints closer than this count as the same pose")
    parser.add_argument("--duplicate_meters", type = float, default = 0.02, help = "and the block closer than this")
    parser.add_argument("--env_config", type = str, default = DEFAULT_ENV_CONFIG, help = "environment config")
    main(parser.parse_args())
