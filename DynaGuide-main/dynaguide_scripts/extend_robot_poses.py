import argparse
import itertools
import json
import os
import sys

import h5py
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import COMPLETED
from core.calvin_utils import BLOCK_INDEX, HIDDEN_POS, STATE_WORDS
from core.guidance_conditions import (BLOCK_POSITION, GRIPPER_WIDTH, Recording, Segment, _anchor_position, find_frames, read_recordings,
                                      save_preview, segment_recording, select_frames, write_conditions)
from extract_block_poses import (ARM_JOINTS, KINDS, base_set_folder, build_base_set, find_duplicate, fixed_joints, frame_kinds, only_block,
                                 pose_in_scene)
from recombine_conditions import (ANCHOR, DEFAULT_CONDITIONS, DEFAULT_ENV_CONFIG, GOALS, LAMPS, STATE_INDEX, Simulator, file_name, held_joints,
                                  make_env, read_seeds, recombine, save_overview)
from scene_tools import block_area, joint_combinations, read_condition_poses, write_rendered

"""
Makes the robot_poses collection bigger: takes the poses it already has, adds robot poses from more recordings (like a part of the CALVIN
D split) until every set has --n different poses, and renders all of them in every scene of their set.

    python dynaguide_scripts/extend_robot_poses.py --recordings datasets/D_Split/task_D_D/training datasets/D_Split/task_D_D/validation \
        --output_folder datasets/robot_poses_new

Run it from DynaGuide-main. Writes the same folders as recombine_conditions.py --hide_blocks and extract_block_poses.py:
    articulated/<behavior>/<scene>.hdf5         every pose in every combination of the other joints, without blocks
    articulated/poses/<behavior>.hdf5           the same poses as they were recorded: the original image and scene, nothing rendered again.
                                                This is what the paper uses as guidance conditions
    blocks/<kind>/<action>_<side>/<scene>.hdf5  and blocks/poses/<kind>.hdf5, see extract_block_poses.py
    manifest.json                               where every pose comes from, how many were found and why candidates were left out

Which poses:
- articulated: the last frame of every segment of a behavior (cut like the authors did). For the button the frames where the gripper
  presses it down, used for button_on and button_off. The joint must be at its end (COMPLETED) and the gripper within --anchor_radius
  of it, because a segment sometimes ends later, when the robot is already somewhere else
- blocks: the frames of extract_block_poses.py (frame_kinds). First one frame per grip or release, frames further back in the
  same one only if that isn't enough
- the poses that are there already come first, then the new ones in a random order (fixed seed)
- a pose like one that is already in the set (every arm joint within 0.05 rad, for blocks also the block within 2 cm) is left out
- poses that fit into every scene of the set come first, so that the set is a full table of poses x scenes. Others are only used if there
  aren't enough, and are missing in the scenes where they don't fit
- a set with more than --n poses is cut to --n, spread evenly over them
"""

BUTTON_FRAMES = {"button" : "pressed", "tcp_near" : "button"} # calvin_exp_configs_examples/guidance_goals/button_pressed.json
DUPLICATE_RADIANS, DUPLICATE_METERS = 0.05, 0.02


def spread(items, n):
    """n of the items, evenly spread."""
    return items if len(items) <= n else [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def without_blocks(states):
    states = states.copy()
    for name, index in BLOCK_INDEX.items():
        states[index : index + 3], states[index + 3 : index + 6] = HIDDEN_POS[name], 0.0
    return states

# --- drawer, door, switch and button ----------------------------------------------------------------------------------------------------

def existing_articulated(paths):
    """The poses of the condition files the base sets were made from, with their original images. {behavior : [pose]}"""
    found = dict()
    for path in paths:
        seeds = read_seeds(path)
        with h5py.File(path, "r") as file:
            demos = sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1]))
            for seed, demo in zip(seeds, demos):
                images = {camera : file["data"][demo]["obs"][camera][-1] for camera in ["third_person", "eye_in_hand"]}
                found.setdefault(seed["behavior"], list()).append({**seed, "images" : images, "origin" : "existing"})
    return found


def at_its_end(behavior, state):
    """Whether the joint of the behavior is where it counts as finished."""
    joint, end = GOALS[behavior]
    value = state[STATE_INDEX[joint]]
    return behavior not in COMPLETED or (value > COMPLETED[behavior] if end == "high" else value < COMPLETED[behavior])


def new_articulated(recordings, anchor_radius):
    """({behavior : [pose]}, {behavior : counts}) of the recordings."""
    found, counts = {behavior : list() for behavior in GOALS}, {behavior : dict(matching = 0, unfinished = 0, robot_away = 0) for behavior in GOALS}

    def add(recording, frame, behavior):
        counts[behavior]["matching"] += 1
        state = recording.states[frame]
        if not at_its_end(behavior, state):
            counts[behavior]["unfinished"] += 1
            return
        anchor = _anchor_position(ANCHOR[GOALS[behavior][0]], state[None])[0]
        if np.linalg.norm(recording.proprio[frame, 0:3] - anchor) > anchor_radius:
            counts[behavior]["robot_away"] += 1
            return
        images = {camera : recording.load_images(camera, [frame])[0] for camera in recording.cameras}
        found[behavior].append({"behavior" : behavior, "source" : f"{recording.name}:{frame}", "states" : state.copy(), "images" : images,
                                "proprio" : recording.proprio[frame].copy(), "action" : recording.actions[frame].copy(), "origin" : "new"})

    for recording in recordings:
        for segment in segment_recording(recording, keep_unfinished = False):
            if segment.behavior in GOALS and GOALS[segment.behavior][0] not in LAMPS:
                add(recording, segment.last, segment.behavior)
    for recording, frame in select_frames(recordings, [find_frames(recording, BUTTON_FRAMES) for recording in recordings]):
        for behavior in ["button_on", "button_off"]:
            add(recording, frame, behavior)
    return found, counts


def same_arm(pose, others):
    return any(np.abs(pose["proprio"][ARM_JOINTS] - other["proprio"][ARM_JOINTS]).max() < DUPLICATE_RADIANS for other in others)


def build_articulated(sim, existing, new, counts, n, rng, folder):
    report = dict()
    for behavior in GOALS:
        goal_joint = GOALS[behavior][0]
        others = [joint for joint in STATE_INDEX if joint != goal_joint]
        combinations = [dict(zip(others, words)) for words in itertools.product(*[sorted(STATE_WORDS[joint]) for joint in others])]
        candidates = list(existing.get(behavior, [])) + [new[behavior][i] for i in rng.permutation(len(new[behavior]))]
        chosen, duplicates, misfits = list(), 0, list()
        for pose in candidates:
            if len(chosen) == n:
                break
            if same_arm(pose, [other for other, _ in chosen]):
                duplicates += 1
                continue
            seed = {**pose, "states" : without_blocks(pose["states"])}
            frozen = [goal_joint] + held_joints(seed, others, 0.1)
            scenes = [recombine(sim, seed, combination, frozen) for combination in combinations]
            reasons = [notes for images, _, notes in scenes if images is None]
            if len(reasons) > 0:
                misfits.append({"source" : pose["source"], "scenes" : len(reasons), "reason" : reasons[0]})
                continue
            chosen.append((pose, scenes))

        os.makedirs(os.path.join(folder, behavior))
        originals = [Segment(recording_of(pose["source"], pose["states"], pose["proprio"], pose["action"], pose["images"]), 0, 0, behavior)
                     for pose, _ in chosen]
        write_conditions(os.path.join(folder, "poses", behavior + ".hdf5"), originals)
        save_preview(os.path.join(folder, "poses", behavior + ".png"), originals)
        for column, combination in enumerate(combinations):
            name = file_name(behavior, combination)
            segments = [Segment(recording_of(pose["source"], scenes[column][1], pose["proprio"], pose["action"], scenes[column][0]), 0, 0, behavior)
                        for pose, scenes in chosen]
            write_conditions(os.path.join(folder, behavior, name + ".hdf5"), segments)
            save_preview(os.path.join(folder, behavior, name + ".png"), segments)
        save_overview(os.path.join(folder, behavior, f"{behavior}_all_no_blocks.png"),
                      [[images["third_person"] for images, _, _ in scenes] for _, scenes in chosen], [file_name(behavior, c) for c in combinations])
        kept = sum(pose["origin"] == "existing" for pose, _ in chosen)
        report[behavior] = {"poses" : len(chosen), "scenes" : len(combinations), "existing" : kept, "new" : len(chosen) - kept,
                            "recordings" : {**counts[behavior], "usable" : len(new[behavior])}, "duplicates" : duplicates, "does_not_fit" : misfits,
                            "sources" : [pose["source"] for pose, _ in chosen]}
        print_row("articulated/" + behavior, report[behavior], n)
    return report


def recording_of(source, states, proprio, action, images):
    return Recording(source, states[None], proprio[None], action[None], list(images),
                     lambda camera, frames, images = images: np.stack([images[camera] for _ in frames]))

# --- blocks --------------------------------------------------------------------------------------------------------------------------------

def existing_blocks(folder):
    """{kind : [pose]} from <folder>/poses/<kind>.hdf5."""
    found = {kind : list() for kind in KINDS}
    for kind in KINDS:
        path = os.path.join(folder, "poses", kind + ".hdf5")
        for pose in read_condition_poses(path) if os.path.exists(path) else []:
            color = str(pose["attrs"]["block"])
            found[kind].append(block_pose(pose["states"], pose["proprio"], pose["action"], color, kind, str(pose["attrs"]["source"]), "existing"))
    return found


def block_pose(states, proprio, action, color, kind, source, origin):
    index = BLOCK_POSITION[color]
    states = only_block(states, color)
    pose = {"states" : states, "proprio" : proprio, "action" : action, "block" : color, "place" : kind.split("_", 1)[1], "origin" : origin,
            "block_position" : states[index : index + 3], "area" : block_area(color, states)}
    pose["info"] = {"source" : source, "block" : color, "area" : pose["area"], "drawer" : round(float(states[1]), 4),
                    "tcp" : [round(float(x), 4) for x in proprio[0:3]], "gripper_width" : round(float(proprio[GRIPPER_WIDTH]), 4)}
    return pose


def new_blocks(recordings, min_gap, rng):
    """{kind : [pose]} in the order to try them: one frame per grip or release in random order, then frames further back in them."""
    found = {kind : list() for kind in KINDS}
    for recording in recordings:
        for (kind, color), mask in frame_kinds(recording.states, recording.proprio).items():
            frames = np.flatnonzero(mask)
            for run in np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1) if len(frames) > 0 else []:
                for tier, position in enumerate(range(len(run) - 1, -1, -min_gap)):
                    frame = int(run[position])
                    found[kind].append((tier, block_pose(recording.states[frame], recording.proprio[frame], recording.actions[frame], color,
                                                         kind, f"{recording.name}:{frame}", "new")))
    for kind, candidates in found.items():
        candidates = [candidates[i] for i in rng.permutation(len(candidates))]
        found[kind] = [pose for _, pose in sorted(candidates, key = lambda candidate: candidate[0])] # stable, so random within a tier
    return found


def build_blocks(sim, existing, new, n, folder):
    report, all_chosen = dict(), {kind : list() for kind in KINDS}
    for kind in KINDS:
        for area in sorted({pose["area"] for pose in existing[kind] + new[kind] if pose["area"] is not None}):
            fixed = fixed_joints(kind, area)
            combinations = joint_combinations(fixed)
            kept, partly, duplicates, tried = list(), list(), 0, 0
            for origin, poses in [("existing", existing[kind]), ("new", new[kind])]:
                for pose in poses:
                    if pose["area"] != area:
                        continue
                    if origin == "new" and len(kept) >= n:
                        break
                    tried += 1
                    if find_duplicate(pose, kept + [other for other, _ in partly], DUPLICATE_RADIANS, DUPLICATE_METERS) is not None:
                        duplicates += 1
                        continue
                    misfits = sum(pose_in_scene(sim, pose, fixed, combination)[1] is not None for combination in combinations)
                    if misfits == 0:
                        kept.append(pose)
                    elif misfits < len(combinations):
                        partly.append((pose, misfits))
            kept = spread(kept, n)
            full = len(kept)
            partly = [pose for pose, _ in sorted(partly, key = lambda entry: entry[1])]
            chosen = kept + partly[: n - full]
            if len(chosen) == 0:
                continue
            set_folder = base_set_folder(folder, kind, area)
            built = build_base_set(sim, chosen, kind, area, set_folder)
            name = os.path.relpath(set_folder, folder)
            old = sum(pose["origin"] == "existing" for pose in chosen)
            report[name] = {"poses" : len(chosen), "scenes" : len(built["scenes"]), "existing" : old, "new" : len(chosen) - old, "in_every_scene" : full,
                            "tried" : tried, "duplicates" : duplicates, "conditions_per_scene" : built["scenes"],
                            "sources" : [pose["info"]["source"] for pose in chosen]}
            print_row("blocks/" + name, report[name], n)
            all_chosen[kind] += chosen
    for kind, poses in all_chosen.items():
        if len(poses) == 0:
            continue
        rendered = list()
        for pose in poses:
            sim.set(pose["states"], pose["proprio"])
            rendered.append((sim.render(), pose["states"], {**pose, "attrs" : {"source" : pose["info"]["source"]}, "file" : pose["info"]["source"]}))
        os.makedirs(os.path.join(folder, "poses"), exist_ok = True)
        segments = write_rendered(os.path.join(folder, "poses", kind + ".hdf5"), rendered, kind,
                                  [{"block" : pose["block"], "place" : pose["place"], "area" : pose["area"]} for pose in poses])
        save_preview(os.path.join(folder, "poses", kind + ".png"), segments)
    return report


def print_row(name, entry, n):
    print(f"{name:42s} {entry['poses']:3d} poses x {entry['scenes']:2d} scenes   {entry['existing']:2d} existing + {entry['new']:2d} new"
          + ("" if entry["poses"] >= n else f"   <- fewer than {n}"), flush = True)


def main(args):
    if os.path.isdir(args.output_folder) and len(os.listdir(args.output_folder)) > 0:
        sys.exit(f"Output folder {args.output_folder} is not empty, choose a new folder so nothing gets overwritten")
    rng = np.random.default_rng(args.seed)
    print("Reading the recordings, a few minutes", flush = True)
    recordings = list()
    for path in args.recordings: # named task_D_D/training_ep0, because the debug set has a training_ep0 as well
        for recording in read_recordings([path]):
            recording.name = f"{os.path.basename(os.path.dirname(os.path.abspath(path)))}/{recording.name}"
            recordings.append(recording)
    print(f"{len(recordings)} recordings with {sum(len(recording) for recording in recordings)} frames", flush = True)
    sim = Simulator(make_env(args.env_config))
    manifest = {"recordings" : args.recordings, "n" : args.n, "anchor_radius" : args.anchor_radius, "seed" : args.seed}

    new, counts = new_articulated(recordings, args.anchor_radius)
    manifest["articulated"] = build_articulated(sim, existing_articulated(args.existing_conditions), new, counts, args.n, rng,
                                                os.path.join(args.output_folder, "articulated"))
    manifest["blocks"] = build_blocks(sim, existing_blocks(args.existing_blocks), new_blocks(recordings, args.min_gap, rng), args.n,
                                      os.path.join(args.output_folder, "blocks"))
    with open(os.path.join(args.output_folder, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent = 2)
    print(f"Saved to {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type = str, nargs = "+", required = True, help = "CALVIN splits (folders with episode_*.npz) to take new poses from")
    parser.add_argument("--output_folder", type = str, required = True, help = "new folder for the bigger collection")
    parser.add_argument("--existing_conditions", type = str, nargs = "*", default = DEFAULT_CONDITIONS, help = "what the articulated sets were made from")
    parser.add_argument("--existing_blocks", type = str, default = os.path.join("datasets", "robot_poses", "blocks"), help = "block sets with their poses folder")
    parser.add_argument("--n", type = int, default = 20, help = "poses per set")
    parser.add_argument("--anchor_radius", type = float, default = 0.12, help = "the gripper must be this close to its joint, in meters")
    parser.add_argument("--min_gap", type = int, default = 8, help = "frames between two candidates of one grip or release")
    parser.add_argument("--seed", type = int, default = 0, help = "order in which new poses are tried")
    parser.add_argument("--env_config", type = str, default = DEFAULT_ENV_CONFIG, help = "environment config")
    main(parser.parse_args())
