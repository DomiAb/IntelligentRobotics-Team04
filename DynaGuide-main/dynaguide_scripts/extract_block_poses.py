import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import COMPLETED
from core.calvin_utils import BLOCK_INDEX, CABINET_Z, DRAWER_Z, HIDDEN_POS, STATE_WORDS, TABLE_Z
from core.guidance_conditions import (BLOCK_POSITION, CABINET_FRONT_Y, GRIPPER_ACTION, GRIPPER_WIDTH, HELD_DISTANCE, HELD_MIN_WIDTH, HELD_WIDTH,
                                      Recording, Segment,
                                      read_calvin_folder, save_preview, write_conditions)
from recombine_conditions import DEFAULT_ENV_CONFIG, DRAWER_BLOCK, STATE_INDEX, Simulator, make_env
from scene_tools import (block_area, held_joint_moved, hide_block, joint_combinations, move_block, new_contacts, save_rows, scene_name,
                         write_rendered)

"""
Collects the robot poses of the standard block interactions for the robot_poses collection: gripping a block and letting go of it, on
the table, in the drawer and in the cabinet. make_task.py puts them into the scene of a task, in the color of its block.

    python dynaguide_scripts/extract_block_poses.py --sources datasets/calvin_debug_dataset/validation datasets/calvin_debug_dataset/training \
        datasets/guidance_conditions results --output_folder datasets/robot_poses/blocks

Run it from DynaGuide-main. A source is a split of a CALVIN dataset (episode_*.npz), an hdf5 file in the robomimic format (rollouts of
run_dynaguide.py, guidance conditions) or a folder, which is searched for hdf5 files. Images aren't needed, the states are enough.
Files that can't be read, like the rollout file of a run that was stopped, are skipped and listed in the manifest.

The kinds, each frame of every recording is checked for each block:
    grip_<place>      the gripper is closed on the block (gripper width 0.03 to 0.06: on a 4 cm block, not closed on nothing next to
                      it), the block lay still on <place> within the last 15 frames and is still within 3 cm of where it lay
    release_<place>   the block lies on <place>, the gripper held it within the last 15 frames and is open now (wider than 0.06),
                      the tcp above the block and within 12 cm of it
    <place> is table, drawer (the drawer open at least COMPLETED["drawer_open"]) or cabinet (the shelf behind the sliding door)

All three blocks are 4 cm wide and 4 cm high, only their length differs (blue 4, red 5.6, pink 8 cm). The gripper always closes on two
4 cm faces, with the long side along the gripper, so a pose of any block works with any other block. That's why the poses are collected
for all blocks together, and duplicates are dropped regardless of color.

Which frames are kept:
- of each run of matching frames, the last frame and then every --min_gap frames further back, because neighbouring frames are the same
- a pose that is the same as one already kept is dropped: same kind, every arm joint within --duplicate_radians and the block within
  --duplicate_meters. The sources are read in the order given, so the debug set comes before the conditions that were cut from it

The output folder:
    poses/<kind>.hdf5 (.png)        the poses as found, one demo each, only the block of the pose in the scene, joints as recorded.
                                    Demo attributes: "block" (its color), "place" and "area"
    <kind>/<action>_<side>/         the base sets, like robot_poses/articulated: every pose with the blue block in place of its own, in
        <scene>.hdf5 (.png)         every combination of drawer, door, switch and green light, one file per scene named like
        <action>_<side>_all.png     drawer_open__door_left__switch_off__green_light_off. <side> is left / center / right of the table
                                    (its thirds) and left / right of the cabinet; the drawer has no sides, its set is <kind>/ itself and
                                    keeps the drawer of each pose. Cabinet sets only have the door where their compartment is open.
                                    Poses that don't fit a scene are left out of it
    manifest.json                   every pose with where it came from, what was dropped and why, and the scenes of every base set
"""

PLACES = ["table", "drawer", "cabinet"]
KINDS = [f"{action}_{place}" for action in ["grip", "release"] for place in PLACES]
WINDOW = 15 # frames to look back for the place the block lay on, or for the grip before a release
GRIP_MOVE = 0.03 # a grip ends when the block is this far from where it lay
GRIPPED_WIDTH = (HELD_MIN_WIDTH, HELD_WIDTH) # gripper width when it holds a 4 cm block
RELEASED_WIDTH = 0.06
RELEASED_TCP_RADIUS = 0.12
REST_TOLERANCE = 0.015
ARM_JOINTS = slice(7, 14) # in proprio
BASE_COLOR = "blue" # the base sets show every pose with this block, make_task.py changes it to the block of the task


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


def resting_place(position, drawer):
    """Per frame the place the block lies on: table, drawer, cabinet, or "" (in the air, or a drawer that isn't open far enough)."""
    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    place = np.full(len(position), "", dtype = object)
    on_top = np.abs(z - TABLE_Z) < REST_TOLERANCE
    place[on_top & (y <= CABINET_FRONT_Y)] = "table"
    place[(np.abs(z - CABINET_Z) < REST_TOLERANCE) & (y > CABINET_FRONT_Y)] = "cabinet"
    in_drawer = (np.abs(z - DRAWER_Z) < REST_TOLERANCE) & (x > DRAWER_BLOCK["x"][0]) & (x < DRAWER_BLOCK["x"][1])
    place[in_drawer & (drawer >= COMPLETED["drawer_open"])] = "drawer"
    return place


def frame_kinds(states, proprio):
    """{(kind, color) : mask over the frames}."""
    masks = dict()
    tcp, width, gripper_action = proprio[:, 0:3], proprio[:, GRIPPER_WIDTH], proprio[:, GRIPPER_ACTION]
    count = len(states)
    for color, index in BLOCK_POSITION.items():
        position = states[:, index : index + 3]
        if position[0, 0] > 2.0: # hidden, out of the scene
            continue
        distance = np.linalg.norm(tcp - position, axis = 1)
        held = (distance < HELD_DISTANCE) & (width > GRIPPED_WIDTH[0]) & (width < GRIPPED_WIDTH[1]) & (gripper_action < 0)
        place = resting_place(position, states[:, 1])
        free = ~held & (place != "")
        grip = {name : np.zeros(count, dtype = bool) for name in PLACES}
        release = {name : np.zeros(count, dtype = bool) for name in PLACES}
        for t in range(1, count):
            before = slice(max(0, t - WINDOW), t)
            if held[t]:
                lay = np.flatnonzero(free[before])
                if len(lay) > 0:
                    last = before.start + lay[-1]
                    if np.linalg.norm(position[t] - position[last]) < GRIP_MOVE:
                        grip[place[last]][t] = True
            elif place[t] != "" and held[before].any() and width[t] > RELEASED_WIDTH and distance[t] < RELEASED_TCP_RADIUS \
                    and tcp[t, 2] > position[t, 2]:
                release[place[t]][t] = True
        for name in PLACES:
            masks[(f"grip_{name}", color)] = grip[name]
            masks[(f"release_{name}", color)] = release[name]
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


def only_block(states, color):
    """The state with the other two blocks hidden."""
    states = states.copy()
    for name, index in BLOCK_INDEX.items():
        if name != color:
            states[index : index + 3], states[index + 3 : index + 6] = HIDDEN_POS[name], 0.0
    return states


def main(args):
    if os.path.isdir(args.output_folder) and len(os.listdir(args.output_folder)) > 0:
        sys.exit(f"Output folder {args.output_folder} is not empty, choose a new folder so nothing gets overwritten")
    recordings, unreadable = read_sources(args.sources)
    print(f"{len(recordings)} recordings, {len(unreadable)} files couldn't be read")

    kept = {kind : list() for kind in KINDS}
    dropped, found = list(), {kind : 0 for kind in KINDS}
    for name, states, proprio, actions in recordings:
        for (kind, color), mask in frame_kinds(states, proprio).items():
            for frame in candidate_frames(mask, args.min_gap):
                index = BLOCK_POSITION[color]
                pose = {"states" : only_block(states[frame], color), "proprio" : proprio[frame], "action" : actions[frame], "block" : color,
                        "place" : kind.split("_", 1)[1], "block_position" : states[frame, index : index + 3],
                        "info" : {"source" : f"{name}:{frame}", "block" : color, "drawer" : round(float(states[frame, 1]), 4),
                                  "block_position" : [round(float(x), 4) for x in states[frame, index : index + 3]],
                                  "tcp" : [round(float(x), 4) for x in proprio[frame, 0:3]],
                                  "gripper_width" : round(float(proprio[frame, GRIPPER_WIDTH]), 4)}}
                found[kind] += 1
                same = find_duplicate(pose, kept[kind], args.duplicate_radians, args.duplicate_meters)
                if same is not None:
                    dropped.append({**pose["info"], "kind" : kind, "reason" : f"same pose as {same}"})
                    continue
                kept[kind].append(pose)

    sim = Simulator(make_env(args.env_config))
    os.makedirs(os.path.join(args.output_folder, "poses"), exist_ok = True)
    manifest = {"sources" : args.sources, "min_gap" : args.min_gap, "duplicate_radians" : args.duplicate_radians,
                "duplicate_meters" : args.duplicate_meters, "unreadable" : unreadable, "poses" : dict(), "dropped" : dropped,
                "base_sets" : dict()}
    for kind in KINDS:
        if len(kept[kind]) == 0:
            print(f"{kind}: no poses found")
            continue
        for pose in kept[kind]:
            pose["area"] = block_area(pose["block"], pose["states"])
            pose["info"]["area"] = pose["area"]
        rendered = list()
        for pose in kept[kind]:
            sim.set(pose["states"], pose["proprio"])
            rendered.append((sim.render(), pose["states"], {**pose, "attrs" : {"source" : pose["info"]["source"]}, "file" : pose["info"]["source"]}))
        path = os.path.join(args.output_folder, "poses", kind + ".hdf5")
        segments = write_rendered(path, rendered, kind, [{"block" : pose["block"], "place" : pose["place"], "area" : pose["area"]}
                                                         for pose in kept[kind]])
        save_preview(os.path.join(args.output_folder, "poses", kind + ".png"), segments)
        manifest["poses"][kind] = [pose["info"] for pose in kept[kind]]
        colors = {color : sum(pose["block"] == color for pose in kept[kind]) for color in BLOCK_POSITION}
        areas = {area : sum(pose["area"] == area for pose in kept[kind]) for area in sorted({pose["area"] for pose in kept[kind]})}
        print(f"{kind}: {len(kept[kind])} poses from {found[kind]} matching frames, by block {colors}, by area {areas}")

    print("Base sets: every pose with the blue block, in every combination of the joints")
    for kind in KINDS:
        for area in sorted({pose["area"] for pose in kept[kind]}):
            poses = [pose for pose in kept[kind] if pose["area"] == area]
            folder = base_set_folder(args.output_folder, kind, area)
            manifest["base_sets"][os.path.relpath(folder, args.output_folder)] = build_base_set(sim, poses, kind, area, folder)

    with open(os.path.join(args.output_folder, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent = 4)
    print(f"Saved to {args.output_folder}")


def base_set_folder(output_folder, kind, area):
    """blocks/grip_cabinet/grip_left for kind grip_cabinet and area cabinet/left, blocks/grip_drawer for the drawer."""
    action = kind.split("_")[0]
    side = area.split("/")[1] if "/" in area else None
    return os.path.join(output_folder, kind, f"{action}_{side}") if side else os.path.join(output_folder, kind)


# the sliding door on the left leaves the right compartment of the cabinet open, and the other way round
OPEN_COMPARTMENT_DOOR = {"cabinet/left" : "right", "cabinet/right" : "left"}


def fixed_joints(kind, area):
    """The joints of a base set that don't take both states: {joint : word}, None keeps the joint of each pose."""
    return {"drawer" : None} if kind.endswith("drawer") else {"sliding_door" : OPEN_COMPARTMENT_DOOR[area]} if area in OPEN_COMPARTMENT_DOOR else {}


def pose_in_scene(sim, pose, fixed, combination, color = BASE_COLOR):
    """The state of a pose in a scene with its block in color, and why it doesn't fit there (None if it does). Leaves it set in sim."""
    state = pose["states"].copy()
    for joint, word in {**{joint : word for joint, word in fixed.items() if word is not None}, **combination}.items():
        state[STATE_INDEX[joint]] = STATE_WORDS[joint][word]
    state[4] = float(state[STATE_INDEX["switch"]] > sim.switch_threshold) # the lightbulb follows the switch
    if pose["block"] != color:
        move_block(state, pose["states"], pose["block"], color, pose["proprio"][3:6])
        hide_block(state, pose["block"])
    joint = held_joint_moved(pose["states"], pose["proprio"][0:3], state)
    return state, f"the gripper is at the {joint}" if joint is not None else \
        new_contacts(sim, pose["states"], state, pose["proprio"], {color : pose["block"]})


def build_base_set(sim, poses, kind, area, folder, color = BASE_COLOR):
    """
    One condition file per scene: every pose with its block replaced by the blue one, in every combination of the joints. A pose in the
    drawer keeps its drawer (the block lies in it), a pose in the cabinet has the door where its compartment is open. A pose is left
    out of a scene where it doesn't fit: the gripper is at a joint the scene moves, or something would be inside something.
    """
    fixed = fixed_joints(kind, area)
    os.makedirs(folder, exist_ok = True)
    rows, report = list(), {"scenes" : dict(), "skipped" : list()}
    for combination in joint_combinations(fixed):
        rendered, attributes = list(), list()
        for pose in poses:
            state, problem = pose_in_scene(sim, pose, fixed, combination, color)
            if problem is not None:
                report["skipped"].append({"scene" : scene_name(state), "source" : pose["info"]["source"], "reason" : problem})
                continue
            rendered.append((sim.render(), state, {**pose, "attrs" : {"source" : pose["info"]["source"]}, "file" : pose["info"]["source"]}))
            attributes.append({"block" : color, "source_block" : pose["block"], "place" : pose["place"], "area" : pose["area"]})
        if len(rendered) == 0:
            continue
        name = scene_name(rendered[0][1])
        segments = write_rendered(os.path.join(folder, name + ".hdf5"), rendered, kind, attributes)
        save_preview(os.path.join(folder, name + ".png"), segments)
        rows.append((name, [images["third_person"] for images, _, _ in rendered]))
        report["scenes"][name] = len(rendered)
    save_rows(os.path.join(folder, os.path.basename(folder) + "_all.png"), rows)
    counts = list(report["scenes"].values())
    print(f"  {os.path.relpath(folder):55s} {len(poses):3d} poses, {len(counts)} scenes, {min(counts) if counts else 0} to "
          f"{max(counts) if counts else 0} conditions per scene")
    return report


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
