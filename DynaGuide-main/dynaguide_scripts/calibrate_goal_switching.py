import argparse
import glob
import itertools
import json
import os
import time

import h5py
import numpy as np
import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import ARTICULATED, COMPLETED
from core.dynaguide import DEVICE
from core.guidance_conditions import read_recordings, segment_recording
from core.multigoal_guidance import SWITCH_DEFAULTS, SwitchCriterion, embed_states, hold_chunk, soft_min_rows

"""
Measures how well the latent switch of multi-goal guidance (core/multigoal_guidance.py) can tell that a goal was reached, on recordings
the conditions don't come from. Two parts:

compute     embeds the conditions of each behavior and every frame of the evaluation recordings, in both embeddings, and saves the
            raw L2 distances: every frame to every condition, and every condition to every other one (distances.npz)
                observed    DINO patches of the frame against the DINO patches of the last frames of the conditions
                predicted   dynamics model prediction for the frame with the last action held, against the condition embeddings of
                            the guidance
analyze     computes the criteria from the saved distances, so they can be changed without embedding again (calibration.json).
            Each frame is labelled per behavior: reached (the joint is at COMPLETED of the behavior), not reached (still on the start side
            of ARTICULATED), or neither (in between, left out). The contrast of a behavior is its opposite (drawer_close for drawer_open).
            AUC (reached against not reached) of the scores d_goal (threshold), d_goal / spread (relative) and d_goal - d_contrast (contrast).
            Then the SwitchCriterion of the rollouts runs over the recorded behaviors, with the default settings and a sweep: too early,
            missed, or how many frames after the env saw the goal reached it switched, where the joint was then, and the false alarms
            when it runs over the whole recording (switches where the goal isn't reached, restarting after each switch).

    python dynaguide_scripts/calibrate_goal_switching.py --guidance dynamics_model_5900.pth \\
        --conditions datasets/calvin_debug_dataset/training --evaluate datasets/calvin_debug_dataset/validation \\
        --output results/multigoal_calibration

Instead of cutting conditions from recordings, --condition_sets takes existing condition files per behavior, like the recombined ones the
experiments use. --exclude_sources leaves out conditions made from the evaluation recordings, which would make the switch look better
than it is. dynaguide_scripts/render_recording_in_sim.py renders the evaluation recordings in the simulator, like the rollouts see them:

    python dynaguide_scripts/calibrate_goal_switching.py --guidance dynamics_model_5900.pth \\
        --condition_sets "drawer_open=datasets/guidance_conditions/recombined_debug/drawer_open/*.hdf5" ... \\
        --exclude_sources validation --evaluate datasets/calvin_debug_dataset_sim/validation.hdf5 --output results/multigoal_calibration_sim

--analyze_only recomputes calibration.json from the distances.npz of the output folder.
"""

MAIN_CAMERA = "third_person"
KINDS = ["observed", "predicted"]
OPPOSITE = {"drawer_open" : "drawer_close", "drawer_close" : "drawer_open", "door_left" : "door_right", "door_right" : "door_left",
            "switch_on" : "switch_off", "switch_off" : "switch_on"}
JOINT_OF = {rising : (idx, low, high, True) for idx, low, high, rising, falling in ARTICULATED}
JOINT_OF.update({falling : (idx, low, high, False) for idx, low, high, rising, falling in ARTICULATED})

# --- compute ------------------------------------------------------------------------------------------------------------------------

def frame_state(recording, frame, image):
    return {MAIN_CAMERA : np.transpose(image, (2, 0, 1)).astype(np.float32), "proprio" : recording.proprio[frame].astype(np.float32)}


def spread_out(items, limit):
    if limit is None or len(items) <= limit:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, limit).round().astype(int)]


def conditions_from_recordings(recordings, behaviors, limit):
    """The conditions of each behavior, made like build_behavior_conditions does: the last frame of every segment of that behavior."""
    found = {behavior : list() for behavior in behaviors}
    for recording in recordings:
        for segment in segment_recording(recording):
            if segment.behavior in found:
                found[segment.behavior].append(segment)
    conditions = dict()
    for behavior, segments in found.items():
        segments = spread_out(segments, limit)
        conditions[behavior] = [(frame_state(s.recording, s.last, s.recording.load_images(MAIN_CAMERA, [s.last])[0]),
                                 hold_chunk(s.recording.actions[s.last], 16), s.label) for s in segments] # as get_labeled_item gives it
    return conditions


def conditions_from_files(specs, behaviors, exclude, limit):
    """Conditions from files: "behavior=pattern" each, the last frame of every demo whose source doesn't contain a word of exclude."""
    files = dict()
    for spec in specs:
        behavior, pattern = spec.split("=", 1)
        files.setdefault(behavior, []).extend(sorted(glob.glob(pattern)))
    conditions = dict()
    for behavior in behaviors:
        found = list()
        for path in files.get(behavior, []):
            with h5py.File(path, "r") as file:
                for demo in sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1])):
                    group = file["data"][demo]
                    source = str(group.attrs.get("source", f"{path}:{demo}"))
                    if any(word in source for word in exclude):
                        continue
                    state = {MAIN_CAMERA : np.transpose(group["obs"][MAIN_CAMERA][-1], (2, 0, 1)).astype(np.float32),
                             "proprio" : group["obs/proprio"][-1].astype(np.float32)}
                    found.append((state, hold_chunk(group["actions"][-1], 16), f"{os.path.basename(path)}:{source}"))
        conditions[behavior] = spread_out(found, limit)
    return conditions


def compute(args):
    torch.set_grad_enabled(False)
    from core.dynamics_models import FinalStatePredictionDino
    model = FinalStatePredictionDino(7, 16, cameras = [MAIN_CAMERA], reconstruction = True, proprio = "proprio", proprio_dim = 15)
    model.load_state_dict(torch.load(args.guidance, map_location = args.device))
    model.to(args.device).eval()

    behaviors = sorted(set(args.behaviors) | {OPPOSITE[b] for b in args.behaviors})
    if args.condition_sets is not None:
        print("Conditions from", args.condition_sets, "without sources containing", args.exclude_sources)
        conditions = conditions_from_files(args.condition_sets, behaviors, args.exclude_sources, args.max_conditions)
    else:
        print("Conditions from", args.conditions)
        conditions = conditions_from_recordings(read_recordings(args.conditions), behaviors, args.max_conditions)
    embedded = dict()
    for behavior in behaviors:
        if len(conditions[behavior]) == 0:
            raise ValueError(f"No conditions for {behavior}")
        states = [state for state, _, _ in conditions[behavior]]
        actions = [chunk for _, chunk, _ in conditions[behavior]]
        embedded[behavior] = {"observed" : embed_states(model, states, None, args.device), "predicted" : embed_states(model, states, actions, args.device)}
        print(f"  {behavior}: {len(states)} conditions")

    saved = {f"sources/{b}" : np.array([label for _, _, label in conditions[b]]) for b in behaviors}
    saved.update({f"proprio/{b}" : np.stack([state["proprio"] for state, _, _ in conditions[b]]) for b in behaviors})
    for kind in KINDS:
        for a in behaviors:
            for b in behaviors:
                saved[f"{kind}/conditions/{a}/{b}"] = torch.cdist(embedded[a][kind], embedded[b][kind]).numpy()

    print("Evaluation frames from", args.evaluate)
    recordings = read_recordings(args.evaluate)
    frame_distances = {kind : {b : list() for b in behaviors} for kind in KINDS}
    all_states, recording_ids, segments = list(), list(), list()
    start = time.time()
    for number, recording in enumerate(recordings):
        frames = np.arange(0, len(recording), args.stride)
        for begin in range(0, len(frames), 64):
            chunk = frames[begin:begin + 64]
            images = recording.load_images(MAIN_CAMERA, chunk)
            states = [frame_state(recording, frame, image) for frame, image in zip(chunk, images)]
            actions = [hold_chunk(recording.actions[max(frame - 1, 0)], 16) for frame in chunk] # the last action, as in a rollout
            queries = {"observed" : embed_states(model, states, None, args.device), "predicted" : embed_states(model, states, actions, args.device)}
            for kind, query in queries.items():
                for behavior in behaviors:
                    frame_distances[kind][behavior].append(torch.cdist(query, embedded[behavior][kind]).numpy())
            print(f"  {recording.name}: {begin + len(chunk)} / {len(frames)} frames, {time.time() - start:.0f} s", flush = True)
        offset = sum(len(s) for s in all_states)
        for segment in segment_recording(recording): # in frames of the evaluation, which keeps every stride-th frame
            segments.append((segment.behavior, offset + segment.first // args.stride, offset + segment.last // args.stride))
        all_states.append(recording.states[frames])
        recording_ids.append(np.full(len(frames), number))
    for kind in KINDS:
        for behavior in behaviors:
            saved[f"{kind}/frames/{behavior}"] = np.concatenate(frame_distances[kind][behavior])
    saved["states"], saved["recording_ids"] = np.concatenate(all_states), np.concatenate(recording_ids)
    saved["segments"] = np.array([f"{b}:{first}:{last}" for b, first, last in segments])
    saved["behaviors"] = np.array(behaviors)
    np.savez_compressed(os.path.join(args.output, "distances.npz"), **saved)

# --- analyze ------------------------------------------------------------------------------------------------------------------------

def frame_labels(states, behavior):
    """1 reached, 0 not reached, -1 in between."""
    idx, low, high, rising = JOINT_OF[behavior]
    value = states[:, idx]
    if rising:
        reached, start_side = value > COMPLETED.get(behavior, high), value < low
    else:
        reached, start_side = value < COMPLETED.get(behavior, low), value > high
    return np.where(reached, 1, np.where(start_side, 0, -1))


def auc(scores, labels):
    """Probability that a reached frame has a lower score (more done) than a not reached one."""
    positive, negative = scores[labels == 1], scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    order = np.argsort(np.concatenate([positive, negative]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float(1 - (ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def episodes_of(data, behavior, after = 60):
    """The recorded behaviors as (frames from the start of the segment to after frames past its end, frame where the goal was reached)."""
    states, recording_ids = data["states"], data["recording_ids"]
    episodes = list()
    for entry in data["segments"]:
        name, first, last = str(entry).rsplit(":", 2)
        first, last = int(first), int(last)
        if name != behavior:
            continue
        same = np.flatnonzero(recording_ids == recording_ids[first])
        window = np.arange(first, min(last + after, same[-1]) + 1)
        labels = frame_labels(states[window], behavior)
        if labels[0] != 0 or not (labels == 1).any(): # starts on the start side and gets there
            continue
        episodes.append((window, int(window[np.argmax(labels == 1)])))
    return episodes


def run_criterion(settings, d_goal, d_contrast, frames, goal_spread = None, rearm = False):
    """
    Runs the SwitchCriterion of the rollouts over recorded distances. Returns the frame of the first switch (None if none), or with rearm
    every switch, starting again after each one like a new goal step.
    """
    criterion = SwitchCriterion(settings, settings["criterion"], goal_spread)
    switches = list()
    for frame in frames:
        if criterion.update(float(d_goal[frame]), float(d_contrast[frame])):
            if not rearm:
                return frame
            switches.append(frame)
            criterion.reset()
    return switches if rearm else None


def scores_of(data, kind, behavior, alpha):
    """d_goal and d_contrast of every frame, and the spread of the goal conditions."""
    contrast = OPPOSITE[behavior]
    d_goal = soft_min_rows(torch.tensor(data[f"{kind}/frames/{behavior}"]), alpha).numpy()
    d_contrast = soft_min_rows(torch.tensor(data[f"{kind}/frames/{contrast}"]), alpha).numpy()
    goal_goal = torch.tensor(data[f"{kind}/conditions/{behavior}/{behavior}"]).clone().fill_diagonal_(float("inf"))
    return d_goal, d_contrast, float(torch.median(goal_goal.min(dim = 1).values))


def switch_outcomes(settings, d_goal, d_contrast, spread, episodes, labels, joint):
    """What a switch with these settings does: on each recorded behavior ('early', 'missed' or the delay after the env saw it done, and
    where the joint was), and over the whole recording, how many of its switches happen where the goal isn't reached (false alarms)."""
    outcomes, positions = list(), list()
    for window, reached in episodes:
        frame = run_criterion(settings, d_goal, d_contrast, window, spread)
        outcomes.append("missed" if frame is None else "early" if frame < reached else int(frame - reached))
        positions.append(None if frame is None else round(float(joint[frame]), 4))
    switches = run_criterion(settings, d_goal, d_contrast, range(len(labels)), spread, rearm = True)
    return {"outcomes" : outcomes, "joint_at_switch" : positions, "switches_whole_recording" : len(switches),
            "false_alarms_whole_recording" : int(sum(labels[f] != 1 for f in switches))}


def analyze(data, alpha = 30, behaviors = None, sweep = None):
    """
    - sweep: {name : list of values} of switch settings to try for the peak criterion, all combinations. By default margin and patience
    """
    behaviors = list(OPPOSITE) if behaviors is None else behaviors
    sweep = {"margin" : [0, 10, 20, 30, 40, 60], "patience" : [3, 5, 8]} if sweep is None else sweep
    states = data["states"]
    results = dict()
    for behavior in behaviors:
        labels = frame_labels(states, behavior)
        episodes = episodes_of(data, behavior)
        joint = states[:, JOINT_OF[behavior][0]]
        entry = {"frames_reached" : int((labels == 1).sum()), "frames_not_reached" : int((labels == 0).sum()), "episodes" : len(episodes),
                 "conditions" : len(data[f"sources/{behavior}"]), "contrast_conditions" : len(data[f"sources/{OPPOSITE[behavior]}"])}
        for kind in KINDS:
            d_goal, d_contrast, goal_spread = scores_of(data, kind, behavior, alpha)
            criteria = {"threshold" : d_goal, "relative" : d_goal / goal_spread, "contrast" : d_goal - d_contrast}
            entry[kind] = {"auc" : {name : auc(score, labels) for name, score in criteria.items()}, "spread" : goal_spread,
                           "default_switch" : switch_outcomes({**SWITCH_DEFAULTS}, d_goal, d_contrast, goal_spread, episodes, labels, joint),
                           "sweep" : list()}
            for values in itertools.product(*sweep.values()):
                settings = {**SWITCH_DEFAULTS, **dict(zip(sweep, values))}
                entry[kind]["sweep"].append({**dict(zip(sweep, values)),
                                             **switch_outcomes(settings, d_goal, d_contrast, goal_spread, episodes, labels, joint)})
        results[behavior] = entry
    return results


def print_results(results):
    for behavior, entry in results.items():
        print(f"{behavior}: {entry['conditions']} conditions, {entry['episodes']} recorded, frames reached {entry['frames_reached']} / not {entry['frames_not_reached']}")
        for kind in KINDS:
            k = entry[kind]
            default = k["default_switch"]
            print(f"   {kind:9s} AUC " + ", ".join(f"{name} {value:.3f}" for name, value in k["auc"].items() if value is not None)
                  + f" | default switch: {default['outcomes']} joint {default['joint_at_switch']}, "
                  f"false alarms {default['false_alarms_whole_recording']} / {default['switches_whole_recording']}")


def main(args):
    os.makedirs(args.output, exist_ok = True)
    if not args.analyze_only:
        compute(args)
    data = dict(np.load(os.path.join(args.output, "distances.npz")))
    results = analyze(data, args.alpha, args.behaviors)
    print_results(results)
    with open(os.path.join(args.output, "calibration.json"), "w") as f:
        json.dump({"settings" : vars(args), "completed" : {b : COMPLETED.get(b) for b in results}, "behaviors" : results}, f, indent = 2)
    print("Wrote", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance", default = None, help = "dynamics model checkpoint")
    parser.add_argument("--conditions", nargs = "+", default = None, help = "recordings the conditions are cut from (CALVIN folder or hdf5)")
    parser.add_argument("--condition_sets", nargs = "+", default = None, help = "or condition files, behavior=pattern each")
    parser.add_argument("--exclude_sources", nargs = "*", default = [], help = "leave out conditions whose source contains one of these")
    parser.add_argument("--evaluate", nargs = "+", default = None, help = "other recordings to evaluate the switch on")
    parser.add_argument("--output", required = True, help = "folder for distances.npz and calibration.json")
    parser.add_argument("--behaviors", nargs = "+", default = list(OPPOSITE), help = "behaviors with an opposite, see OPPOSITE")
    parser.add_argument("--alpha", type = float, default = 30, help = "of the soft minimum, as the guidance uses it")
    parser.add_argument("--stride", type = int, default = 1, help = "evaluate every n-th frame, to save time")
    parser.add_argument("--max_conditions", type = int, default = None, help = "conditions per behavior at most")
    parser.add_argument("--device", default = DEVICE)
    parser.add_argument("--analyze_only", action = "store_true", help = "only recompute calibration.json from distances.npz")
    args = parser.parse_args()
    if not args.analyze_only and ((args.conditions is None) == (args.condition_sets is None) or args.evaluate is None or args.guidance is None):
        parser.error("give --guidance, --evaluate and either --conditions or --condition_sets")
    main(args)
