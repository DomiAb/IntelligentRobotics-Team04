import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import ARTICULATED, COMPLETED, get_behavior_goals, is_success
from core.guidance_conditions import Recording, Segment, save_preview, write_conditions
from recombine_conditions import DEFAULT_ENV_CONFIG, GOALS, STATE_INDEX, Simulator, make_env, read_seeds, recombine

"""
Collects the robot at the end of every successful rollout as a goal robot pose, one file per behavior. Existing condition files
can go in as well (--conditions), to merge all goal robots into one set, e.g. with --completed_only for a set that only shows
finished motions. The files have the format of
the guidance conditions, so recombine_conditions.py takes them with --conditions and renders each pose in every combination of the
other joints, with or without blocks, like the poses of the debug set. See section 7 of "Create guidance conditions.ipynb".

    python dynaguide_scripts/extract_robot_poses.py --results results/experiments_first_1 results/experiments_first_2 \
        --output_folder datasets/robot_poses/first

    python dynaguide_scripts/extract_robot_poses.py --results results/experiments_first_1 ... \
        --conditions datasets/guidance_conditions/debug_all/door_left.hdf5 ... --completed_only --output_folder datasets/robot_poses/ground_truth

Run it from DynaGuide-main. Each results folder is the output of run_experiments.py or harvest_rollouts.py: every <experiment>/<variant>/
with a DONE, a config.json, a *_log.json and a *_rollout.hdf5 is read. The rollouts don't need images, the states are enough.

What a pose is:
- the last step of a successful episode, where the GoalTracker of that run ended it: the scene state (joints and blocks), the robot
  (proprio) and the action of that step
- success as the run scored it, in its *_log.json. A run made before the stricter COMPLETED thresholds ended its episodes earlier, at
  a half open drawer for example. Each pose notes its joint value and whether it reaches COMPLETED, --completed_only keeps only those
- for a condition file (--conditions), the last frame of each demo, which is what the guidance uses. Its behavior comes from the
  demo's "behavior" attribute, only drawer_open, drawer_close, door_left, door_right, switch_on, switch_off, button_on, button_off
- the base policy runs count too, a pose is a pose. --skip_base_policy leaves them out
- conditions come first, so when a rollout ends in the same pose as a condition, the condition is kept
- the same pose twice, e.g. two variants of one experiment that ran the same start scene into the same end, is kept once: same
  behavior, every arm joint within --duplicate_radians and the goal joint within --duplicate_joint

The output folder:
    <behavior>.hdf5     one demo per pose, a single frame, rendered again in the simulator in its own scene (CPU renderer)
    <behavior>.png      the poses as rendered
    manifest.json       every pose with where it came from, its joint value and the COMPLETED check, and what was dropped and why
"""

ARM_JOINTS = slice(7, 14) # in proprio
LOG_SUFFIX, ROLLOUT_SUFFIX = "_log.json", "_rollout.hdf5"
JOINT_OF_BEHAVIOR = {name : idx for idx, _, _, rising, falling in ARTICULATED for name in [rising, falling]}


def find_tasks(results_folder):
    """Every finished task of a results folder: (experiment, variant, folder)."""
    tasks = list()
    for done in sorted(glob.glob(os.path.join(results_folder, "*", "*", "DONE"))):
        folder = os.path.dirname(done)
        tasks.append((os.path.basename(os.path.dirname(folder)), os.path.basename(folder), folder))
    return tasks


def one_file(folder, suffix):
    files = sorted(glob.glob(os.path.join(folder, "*" + suffix)))
    if len(files) != 1:
        raise FileNotFoundError(f"{folder} has {len(files)} *{suffix} files, expected one")
    return files[0]


def read_poses(results_folder, experiment, variant, folder):
    """The last frame of each successful episode of one task."""
    with open(os.path.join(folder, "config.json"), "r") as f:
        targets, avoid = get_behavior_goals(json.load(f))
    with open(one_file(folder, LOG_SUFFIX), "r") as f:
        episodes = json.load(f)["Episode_Behaviors"]
    poses = list()
    with h5py.File(one_file(folder, ROLLOUT_SUFFIX), "r") as file:
        for number, behaviors in enumerate(episodes):
            if not is_success(behaviors, targets, avoid):
                continue
            demo = file["data"][f"demo_{number}"]
            last = demo["next_obs"] if "next_obs" in demo else demo["obs"] # next_obs is the state after the last step
            reached = [name for name in targets if name in behaviors]
            if len(reached) != 1:
                print(f"    skipped {folder} demo_{number}: reached {reached}, can't tell which behavior the pose shows")
                continue
            behavior = reached[0]
            states, start = last["states"][-1].astype(np.float64), demo["obs"]["states"][0]
            joint = JOINT_OF_BEHAVIOR.get(behavior)
            poses.append({"behavior" : behavior, "states" : states, "proprio" : last["proprio"][-1].astype(np.float64),
                          "action" : demo["actions"][-1].astype(np.float64),
                          "info" : {"source" : f"{os.path.basename(os.path.normpath(results_folder))}/{experiment}/{variant}/demo_{number}",
                                    "variant" : variant, "steps" : int(demo.attrs["num_samples"]),
                                    "joint_start" : None if joint is None else round(float(start[joint]), 4),
                                    "joint_end" : None if joint is None else round(float(states[joint]), 4),
                                    "completed" : completed(behavior, states)}})
    return poses


def read_condition_poses(path):
    """The last frame of each demo of a condition file, with the same information as a rollout pose."""
    with h5py.File(path, "r") as file:
        demos = sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1])) # the order of read_seeds
        firsts = [(file["data"][demo]["obs/states"][0], len(file["data"][demo]["actions"])) for demo in demos]
    poses = list()
    for demo, (start, length), seed in zip(demos, firsts, read_seeds(path)):
        joint = JOINT_OF_BEHAVIOR.get(seed["behavior"])
        poses.append({**seed, "info" : {"source" : seed["source"], "variant" : "condition", "file" : f"{path}:{demo}", "steps" : length,
                                        "joint_start" : None if joint is None else round(float(start[joint]), 4),
                                        "joint_end" : None if joint is None else round(float(seed["states"][joint]), 4),
                                        "completed" : completed(seed["behavior"], seed["states"])}})
    return poses


def completed(behavior, states):
    """Whether the pose reaches the COMPLETED value of its behavior, True for behaviors without one like the button."""
    if behavior not in COMPLETED:
        return True
    value, threshold = states[JOINT_OF_BEHAVIOR[behavior]], COMPLETED[behavior]
    return bool(value > threshold if GOALS[behavior][1] == "high" else value < threshold)


def is_duplicate(pose, kept, radians, joint_tolerance):
    joint = STATE_INDEX[GOALS[pose["behavior"]][0]]
    for other in kept:
        if np.abs(pose["proprio"][ARM_JOINTS] - other["proprio"][ARM_JOINTS]).max() < radians \
                and abs(pose["states"][joint] - other["states"][joint]) < joint_tolerance:
            return other["info"]["source"]
    return None


def main(args):
    if os.path.isdir(args.output_folder) and len(os.listdir(args.output_folder)) > 0:
        sys.exit(f"Output folder {args.output_folder} is not empty, choose a new folder so nothing gets overwritten")
    if len(args.results) + len(args.conditions) == 0:
        sys.exit("Nothing to read, give --results and/or --conditions")
    sources = [(path, lambda path = path: read_condition_poses(path)) for path in args.conditions]
    for results_folder in args.results:
        tasks = find_tasks(results_folder)
        if len(tasks) == 0:
            sys.exit(f"{results_folder} has no finished tasks (<experiment>/<variant>/DONE)")
        sources += [(folder, lambda args_ = (results_folder, experiment, variant, folder): read_poses(*args_))
                    for experiment, variant, folder in tasks if not (args.skip_base_policy and variant == "base_policy")]

    by_behavior, dropped = dict(), list()
    for name, read in sources:
        poses = read()
        print(f"{name}: {len(poses)} poses")
        for pose in poses:
            if args.behaviors is not None and pose["behavior"] not in args.behaviors:
                continue
            if args.completed_only and not pose["info"]["completed"]:
                dropped.append({**pose["info"], "reason" : f"doesn't reach COMPLETED {COMPLETED[pose['behavior']]}"})
                continue
            kept = by_behavior.setdefault(pose["behavior"], list())
            same = is_duplicate(pose, kept, args.duplicate_radians, args.duplicate_joint)
            if same is not None:
                dropped.append({**pose["info"], "reason" : f"same pose as {same}"})
                continue
            kept.append(pose)

    sim = Simulator(make_env(args.env_config))
    os.makedirs(args.output_folder, exist_ok = True)
    manifest = {"results" : args.results, "conditions" : args.conditions, "completed_only" : args.completed_only, "skip_base_policy" : args.skip_base_policy,
                "duplicate_radians" : args.duplicate_radians, "duplicate_joint" : args.duplicate_joint, "poses" : dict(), "dropped" : dropped}
    for behavior, poses in sorted(by_behavior.items()):
        segments, entries = list(), list()
        for pose in poses:
            images, states, notes = recombine(sim, pose, dict(), [GOALS[behavior][0]]) # its own scene, nothing changes
            if images is None:
                dropped.append({**pose["info"], "reason" : notes})
                continue
            recording = Recording(pose["info"]["source"], states[None], pose["proprio"][None], pose["action"][None], ["third_person", "eye_in_hand"],
                                  lambda camera, frames, images = images: np.stack([images[camera] for _ in frames]))
            segments.append(Segment(recording, 0, 0, behavior))
            entries.append({**pose["info"], **notes})
        if len(segments) == 0:
            continue
        write_conditions(os.path.join(args.output_folder, behavior + ".hdf5"), segments)
        save_preview(os.path.join(args.output_folder, behavior + ".png"), segments)
        manifest["poses"][behavior] = entries
        done = sum(entry["completed"] for entry in entries)
        print(f"{behavior}: {len(entries)} poses, {done} reach COMPLETED")
    with open(os.path.join(args.output_folder, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent = 4)
    print(f"{len(dropped)} dropped, see manifest.json. Saved to {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type = str, nargs = "*", default = [], help = "results folders of run_experiments.py or harvest_rollouts.py")
    parser.add_argument("--conditions", type = str, nargs = "*", default = [], help = "condition files, the last frame of each demo is a pose")
    parser.add_argument("--output_folder", type = str, required = True, help = "new folder for the pose files")
    parser.add_argument("--behaviors", type = str, nargs = "+", default = None, help = "only these behaviors")
    parser.add_argument("--completed_only", action = "store_true", help = "only poses whose joint reaches COMPLETED")
    parser.add_argument("--skip_base_policy", action = "store_true", help = "leave out the rollouts of the base policy")
    parser.add_argument("--duplicate_radians", type = float, default = 0.02, help = "arm joints closer than this count as the same pose")
    parser.add_argument("--duplicate_joint", type = float, default = 0.005, help = "and the goal joint closer than this")
    parser.add_argument("--env_config", type = str, default = DEFAULT_ENV_CONFIG, help = "environment config")
    main(parser.parse_args())
