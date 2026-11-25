import argparse
import glob
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
from core.multigoal_guidance import CHANGE_DEFAULTS, SWITCH_DEFAULTS, SwitchCriterion, change_score, embed_states
from make_task import before_state
from recombine_conditions import DEFAULT_ENV_CONFIG, GOALS, STATE_INDEX, Simulator, make_env
from scene_tools import held_joint_moved, new_contacts, read_condition_poses

"""
Tests the change criterion of the goal switch (core/multigoal_guidance.py) in three ways, see "Multigoal guidance.ipynb". No rollouts are
run here. The truth is always the joint in the simulator state, which the switch never sees.

live        what the switch did in finished rollouts of run_experiments.py, read from their logs. Nothing is rendered or embedded
                python dynaguide_scripts/test_goal_switch.py live --results results/goal_switch results/goal_switch_previous

replay      runs the switch of every step over recorded rollouts of a made task, every frame rendered again: also over rollouts that used
            the oracle or another switch, where it can be wrong in other ways. Every step's switch runs over the whole episode, also while
            DynaGuide wasn't guiding that step, so this contains many chances to switch wrongly
                python dynaguide_scripts/test_goal_switch.py replay --results results/goal_switch --guidance dynamics_model_5900.pth

recordings  the same for any recordings, without a task: for each recorded behavior the conditions are made in the scene it starts from,
            from the poses of datasets/robot_poses (without the poses that come from the same place), and the switch runs over the
            recording. Other goals that stay unreached run over it as well, they must not switch
                python dynaguide_scripts/test_goal_switch.py recordings --recordings datasets/D_Split/task_D_D/validation \\
                    --guidance dynamics_model_5900.pth --output results/goal_switch_recordings

What is counted. A joint is "reached" past COMPLETED of the behavior (what the env counts as done), "at the start" on the start side of
ARTICULATED, else "on the way". For each step of each episode:
    correct         switched while reached, with the delay in env steps since it was reached
    on the way      switched with the joint between start and reached. The list of joint values shows how far off it was
    false           switched with the joint still at the start. The worst kind
    missed          reached for at least MISSED_AFTER steps, not switched
    too short       reached, but the recording ends sooner than that
    quiet           not reached and not switched, as it should be
    already         reached before the step began (the LED was already on): switched or not, no test
The scores of replay and recordings are saved (scores.npz of --output or the results folder), and --analyze_only evaluates them again for
--margins, --patience and --tolerance.
"""

MAIN_CAMERA = "third_person"
JOINT_OF = {rising : (idx, low, high, True) for idx, low, high, rising, falling in ARTICULATED}
JOINT_OF.update({falling : (idx, low, high, False) for idx, low, high, rising, falling in ARTICULATED})
MISSED_AFTER = 40
OUTCOMES = ["correct", "on the way", "false", "missed", "too short", "quiet", "already"]

# --- the truth ----------------------------------------------------------------------------------------------------------------------

def joint_side(behavior, value):
    """1 where the joint is reached, 0 at the start, -1 on the way."""
    _, low, high, rising = JOINT_OF[behavior]
    value = np.asarray(value)
    if rising:
        reached, start = value > COMPLETED.get(behavior, high), value < low
    else:
        reached, start = value < COMPLETED.get(behavior, low), value > high
    return np.where(reached, 1, np.where(start, 0, -1))


def outcome(side, steps, switched, last_step):
    """
    - side: joint_side at each of steps (env steps, from where the step's switch ran). last_step: the last env step of the episode
    - switched: the env step of the switch, or None
    Returns (outcome, delay or None).
    """
    side, steps = np.asarray(side), np.asarray(steps)
    if side[0] == 1:
        return "already", None
    if switched is None:
        reached = np.flatnonzero(side == 1)
        if len(reached) == 0:
            return "quiet", None
        for first in reached: # a goal that is undone again soon isn't missed
            after = side[(steps >= steps[first]) & (steps <= steps[first] + MISSED_AFTER)]
            if (after == 1).all() and last_step - steps[first] >= MISSED_AFTER:
                return "missed", None
        return "too short", None
    at = int(np.searchsorted(steps, switched))
    if side[at] == 1:
        before = np.flatnonzero(side[:at] != 1)
        return "correct", int(switched - steps[before[-1] + 1])
    return ("false" if side[at] == 0 else "on the way"), None


class Table:
    """Counts the outcomes per (group, behavior) and prints them."""
    def __init__(self):
        self.rows = dict()

    def add(self, group, behavior, result, delay = None, joint = None, label = None):
        row = self.rows.setdefault((group, behavior), {name : 0 for name in OUTCOMES} | {"delays" : [], "joint_on_the_way" : [], "wrong" : []})
        row[result] += 1
        if delay is not None:
            row["delays"].append(delay)
        if result == "on the way":
            row["joint_on_the_way"].append(round(float(joint), 3))
        if result in ["false", "missed"]:
            row["wrong"].append(f"{label} ({result})")

    def totals(self, group = None):
        rows = [row for (g, _), row in self.rows.items() if group is None or g == group]
        return {name : sum(row[name] for row in rows) for name in OUTCOMES}

    def print(self):
        print(f"{'':44s}" + "".join(f"{name:>11s}" for name in OUTCOMES) + "   delay mean / max")
        for (group, behavior), row in self.rows.items():
            delays = f"{np.mean(row['delays']):6.1f} / {max(row['delays'])}" if len(row["delays"]) > 0 else "-"
            print(f"{group + ' ' + behavior:44s}" + "".join(f"{row[name]:11d}" for name in OUTCOMES) + f"   {delays}")
            if len(row["joint_on_the_way"]) > 0:
                print(f"{'':6s}on the way at {row['joint_on_the_way']}, reached is {COMPLETED.get(behavior)}")
            if len(row["wrong"]) > 0:
                print(f"{'':6s}{', '.join(row['wrong'])}")
        print(f"{'all':44s}" + "".join(f"{count:11d}" for count in self.totals().values()))

    def as_dict(self):
        return {f"{group} {behavior}" : row for (group, behavior), row in self.rows.items()}

# --- rollouts of run_experiments.py ------------------------------------------------------------------------------------------------------

def finished_runs(folders):
    """(experiment, variant, config, log, rollout file) of every finished multi-goal task."""
    runs = list()
    for folder in folders:
        for done in sorted(glob.glob(os.path.join(folder, "*", "*", "DONE"))):
            task = os.path.dirname(done)
            with open(os.path.join(task, "config.json"), "r") as f:
                config = json.load(f)
            logs, rollouts = glob.glob(os.path.join(task, "*_log.json")), glob.glob(os.path.join(task, "*_rollout.hdf5"))
            if "goal_sequence" not in config or len(logs) != 1 or len(rollouts) != 1:
                print(f"{task}: not a multi-goal run with one log and one rollout file, skipped")
                continue
            with open(logs[0], "r") as f:
                log = json.load(f)
            runs.append((os.path.basename(os.path.dirname(task)), os.path.basename(task), config, log, rollouts[0]))
    if len(runs) == 0:
        raise SystemExit(f"No finished multi-goal runs in {folders}")
    return runs


def step_behavior(entry):
    """The one joint behavior a goal step asks for, or None (a block step)."""
    names = [name for name in entry.get("behaviors", []) if name in JOINT_OF]
    return names[0] if len(names) == 1 else None


def episodes(rollout_file):
    """The states and proprio of each episode. Entry 0 is the start, entry t the state after env step t, as the switch counts them."""
    with h5py.File(rollout_file, "r") as file:
        for demo in sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1])):
            group = file["data"][demo]
            yield demo, *[np.concatenate([group[f"obs/{key}"][:1], group[f"next_obs/{key}"][()]]) for key in ["states", "proprio"]]


def live(args):
    table, summary = Table(), dict()
    for experiment, variant, config, log, rollout_file in finished_runs(args.results):
        steps = config["goal_sequence"]
        switch_logs = log["Episode_Switches"]
        for number, (demo, states, _) in enumerate(episodes(rollout_file)):
            begin = 0
            for k, entry in enumerate(steps[:-1]): # the last step is ended by the env
                switched = next((s["t"] for s in switch_logs[number]["switches"] if s["from_step"] == k), None)
                behavior = step_behavior(entry)
                if behavior is not None:
                    frames = np.arange(begin, len(states) if switched is None else switched + 1)
                    values = states[frames, JOINT_OF[behavior][0]]
                    result, delay = outcome(joint_side(behavior, values), frames, switched, len(states) - 1)
                    table.add(f"{experiment}/{variant}", behavior, result, delay, values[-1], demo)
                if switched is None:
                    break
                begin = switched
        summary[f"{experiment}/{variant}"] = {"success" : log["Num_Success"], "episodes" : log["Num_Episode"] if "Num_Episode" in log else len(switch_logs),
                                              "step_reach_rate" : log["Step_Reach_Rate"], "outcomes" : table.totals(f"{experiment}/{variant}")}
    print("\nEvery switch of every rollout\n")
    table.print()
    print("\nGoals reached (env) and what the switch did\n")
    for name, row in summary.items():
        rates = ", ".join(f"{step} {rate:.2f}" for step, rate in row["step_reach_rate"].items())
        print(f"{name:44s} success {row['success']:2d}/{row['episodes']:2d}   reached: {rates}")
    output = args.output or args.results[0]
    os.makedirs(output, exist_ok = True)
    path = os.path.join(output, "switch_live.json")
    with open(path, "w") as f:
        json.dump({"results" : args.results, "outcomes" : table.as_dict(), "runs" : summary}, f, indent = 2)
    print(f"\nWrote {path}")

# --- scores over recorded frames -----------------------------------------------------------------------------------------------------

class Embedder:
    def __init__(self, guidance, device):
        from core.dynamics_models import FinalStatePredictionDino
        self.model = FinalStatePredictionDino(7, 16, cameras = [MAIN_CAMERA], reconstruction = True, proprio = "proprio", proprio_dim = 15)
        self.model.load_state_dict(torch.load(guidance, map_location = device))
        self.model.to(device).eval()
        self.device = device

    def __call__(self, images):
        states = [{MAIN_CAMERA : np.transpose(image, (2, 0, 1)).astype(np.float32), "proprio" : np.zeros(15, np.float32)} for image in images]
        return embed_states(self.model, states, None, self.device)


def rendered(sim, states, proprio, frames):
    images = list()
    for frame in frames:
        sim.set(states[frame].astype(np.float64), proprio[frame].astype(np.float64))
        images.append(sim.render()[MAIN_CAMERA])
    return images


def last_images(path):
    with h5py.File(path, "r") as file:
        demos = sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1]))
        return [file["data"][demo]["obs"][MAIN_CAMERA][-1] for demo in demos]


class Scores:
    """The change score of a goal over the frames of an episode, one entry per (episode, goal)."""
    def __init__(self):
        self.entries = list()

    def add(self, group, label, behavior, steps, score, joint, last_step):
        self.entries.append({"group" : group, "label" : label, "behavior" : behavior, "steps" : np.asarray(steps),
                             "score" : np.asarray(score), "joint" : np.asarray(joint), "last_step" : int(last_step)})

    def save(self, path):
        arrays = {"names" : np.array([f"{e['group']}|{e['label']}|{e['behavior']}|{e['last_step']}" for e in self.entries])}
        for number, entry in enumerate(self.entries):
            arrays.update({f"{key}/{number}" : entry[key] for key in ["steps", "score", "joint"]})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        scores = cls()
        with np.load(path) as data:
            for number, name in enumerate(data["names"]):
                group, label, behavior, last_step = str(name).split("|")
                scores.add(group, label, behavior, *[data[f"{key}/{number}"] for key in ["steps", "score", "joint"]], int(last_step))
        return scores


def first_switch(entry, settings):
    """Runs the SwitchCriterion of the rollouts over the scores. patience is in env steps, the frames are every few steps."""
    stride = int(np.median(np.diff(entry["steps"]))) if len(entry["steps"]) > 1 else 1
    criterion = SwitchCriterion({**settings, "patience" : max(1, round(settings["patience"] / stride))}, "change")
    for step, score in zip(entry["steps"], entry["score"]):
        if step - entry["steps"][0] >= settings["min_steps"] and criterion.update_score(float(score)):
            return int(step)
    return None


def evaluate(scores, settings):
    table = Table()
    for entry in scores.entries:
        switched = first_switch(entry, settings)
        side = joint_side(entry["behavior"], entry["joint"])
        result, delay = outcome(side, entry["steps"], switched, entry["last_step"])
        joint = None if switched is None else entry["joint"][np.searchsorted(entry["steps"], switched)]
        table.add(entry["group"], entry["behavior"], result, delay, joint, entry["label"])
    return table


def analyze(scores, args, output):
    if len(scores.entries) == 0:
        raise SystemExit("No scores to evaluate, see the messages above")
    settings = {**SWITCH_DEFAULTS, **CHANGE_DEFAULTS}
    print(f"\nWith the defaults: margin {settings['margin']}, tolerance {settings['tolerance']}, patience {settings['patience']}\n")
    table = evaluate(scores, settings)
    table.print()
    sweep = list()
    print("\nOther settings, all groups and behaviors together\n\nmargin  patience  tolerance" + "".join(f"{name:>11s}" for name in OUTCOMES))
    for patience in args.patience:
        for tolerance in args.tolerance:
            for margin in args.margins:
                totals = evaluate(scores, {**settings, "margin" : margin, "patience" : patience, "tolerance" : tolerance}).totals()
                sweep.append({"margin" : margin, "patience" : patience, "tolerance" : tolerance, **totals})
                print(f"{margin:6.2f}{patience:10d}{tolerance:11.2f}" + "".join(f"{totals[name]:11d}" for name in OUTCOMES))
    print("\nScore of each behavior: lowest 2 % where it was reached, highest 2 % at the start\n")
    separation = dict()
    for behavior in sorted({entry["behavior"] for entry in scores.entries}):
        found = [entry for entry in scores.entries if entry["behavior"] == behavior]
        score, side = np.concatenate([e["score"] for e in found]), np.concatenate([joint_side(behavior, e["joint"]) for e in found])
        row = {"reached_frames" : int((side == 1).sum()), "start_frames" : int((side == 0).sum()),
               "reached_low" : float(np.quantile(score[side == 1], 0.02)) if (side == 1).any() else None,
               "reached_median" : float(np.median(score[side == 1])) if (side == 1).any() else None,
               "start_high" : float(np.quantile(score[side == 0], 0.98)) if (side == 0).any() else None,
               "start_max" : float(score[side == 0].max()) if (side == 0).any() else None}
        separation[behavior] = row
        print(f"{behavior:14s}" + "".join(f"   {key} {value:.2f}" if isinstance(value, float) else f"   {key} {value}" for key, value in row.items()))
    path = os.path.join(output, "switch_replay.json")
    with open(path, "w") as f:
        json.dump({"settings" : {key : settings[key] for key in ["margin", "tolerance", "patience", "min_steps"]}, "totals" : table.totals(),
                   "outcomes" : table.as_dict(), "sweep" : sweep, "separation" : separation}, f, indent = 2)
    print(f"\nWrote {path}")


def replay(args):
    output = args.output or args.results[0]
    scores_file = os.path.join(output, "scores.npz")
    if args.analyze_only:
        return analyze(Scores.load(scores_file), args, output)
    runs = finished_runs(args.results)
    os.makedirs(output, exist_ok = True)
    embed, sim = Embedder(args.guidance, args.device), Simulator(make_env(args.env_config))
    scores, cached, start = Scores(), dict(), time.time()
    for experiment, variant, config, _, rollout_file in runs:
        goals = list()
        for entry in config["goal_sequence"]:
            behavior, examples = step_behavior(entry), entry.get("pos_examples")
            if behavior is None or not isinstance(examples, str):
                print(f"{experiment}: step {entry.get('name')} isn't one joint behavior with one condition file, skipped")
                continue
            examples = examples.replace("\\", "/") # a run from the workstation, looked at on another machine
            files = [os.path.join(args.task or os.path.dirname(examples), os.path.basename(examples)[:-5] + end) for end in [".hdf5", "_before.hdf5"]]
            if not all(os.path.isfile(path) for path in files):
                print(f"{experiment}: step {entry.get('name')} has no {files[1]}, skipped")
                continue
            if files[0] not in cached:
                cached[files[0]] = [embed(last_images(path)) for path in files]
            goals.append((behavior, *cached[files[0]]))
        for number, (demo, states, proprio) in enumerate(episodes(rollout_file)):
            if len(goals) == 0 or (args.max_episodes is not None and number >= args.max_episodes):
                break
            frames = np.arange(0, len(states), args.stride)
            images = rendered(sim, states, proprio, frames)
            parts = {behavior : list() for behavior, _, _ in goals}
            for begin in range(0, len(frames), 16):
                query = embed(images[begin:begin + 16])
                for behavior, goal, before in goals:
                    parts[behavior].append(change_score(goal, before, query).cpu().numpy())
            for behavior, _, _ in goals:
                scores.add(f"{experiment}/{variant}", demo, behavior, frames, np.concatenate(parts[behavior]),
                           states[frames, JOINT_OF[behavior][0]], len(states) - 1)
            print(f"{experiment}/{variant} {demo}: {len(frames)} frames, {time.time() - start:.0f} s", flush = True)
        scores.save(scores_file)
    analyze(scores, args, output)

# --- recordings without a task ----------------------------------------------------------------------------------------------------------

class SceneOnly:
    """What make_task.before_state needs of its story: the scene before the step."""
    def __init__(self, state):
        self.state, self.held, self.block = state, False, None


class Pose:
    def __init__(self, behavior):
        self.behavior, self.joint = behavior, GOALS[behavior][0]


def pairs_in_scene(sim, behavior, poses, scene, exclude, limit, rng):
    """(goal images, before images) of the poses of a behavior, rendered in a scene. exclude: the tcp the poses must not come from."""
    joint, goal_images, before_images = GOALS[behavior][0], list(), list()
    index = STATE_INDEX[joint]
    for number in rng.permutation(len(poses)):
        pose = poses[number]
        if exclude is not None and np.linalg.norm(pose["proprio"][:3] - exclude) < 1e-3:
            continue
        state = scene.copy()
        if joint == "drawer":
            for block in [6, 12, 18]: # a block in the drawer moves with it
                if 0.0 < state[block] < 0.36 and state[block + 2] < 0.40:
                    state[block + 1] -= pose["states"][index] - scene[index]
        state[index] = pose["states"][index]
        if joint == "green_light":
            state[index], state[2] = float(JOINT_OF[behavior][3]), pose["states"][2]
        if held_joint_moved(pose["states"], pose["proprio"][:3], state) is not None or new_contacts(sim, pose["states"], state, pose["proprio"]) is not None:
            continue
        before = before_state(SceneOnly(scene), Pose(behavior), state, None)
        if joint == "drawer":
            before[[7, 13, 19]] = scene[[7, 13, 19]]
        for images, scene_state in [(goal_images, state), (before_images, before)]:
            sim.set(scene_state, pose["proprio"])
            images.append(sim.render()[MAIN_CAMERA])
        if len(goal_images) == limit:
            break
    return goal_images, before_images


def recordings(args):
    scores_file = os.path.join(args.output, "scores.npz")
    if args.analyze_only:
        return analyze(Scores.load(scores_file), args, args.output)
    os.makedirs(args.output, exist_ok = True)
    embed, sim, rng = Embedder(args.guidance, args.device), Simulator(make_env(args.env_config)), np.random.default_rng(0)
    poses = {behavior : read_condition_poses(os.path.join(args.poses, f"{behavior}.hdf5")) for behavior in GOALS}
    found = {behavior : list() for behavior in GOALS}
    paths = [path for pattern in args.recordings for path in (sorted(glob.glob(pattern)) if "*" in pattern else [pattern])]
    for path in paths:
        for recording in read_recordings([path]):
            group = os.path.relpath(path, "results" if path.startswith("results") else "datasets").split(os.sep)[0]
            is_rollout = path.endswith(".hdf5")
            segments = [(0, len(recording) - 1, None)] if is_rollout else [(s.first, s.last, s.behavior) for s in segment_recording(recording, keep_unfinished = False)]
            for first, last, behavior in segments: # a rollout counts for every joint that starts on the start side of its file's goal
                behavior = next((b for b in GOALS if f"{os.sep}{b}{os.sep}" in os.path.normpath(path)), None) if is_rollout else behavior
                if behavior in found and joint_side(behavior, recording.states[first, JOINT_OF[behavior][0]]) == 0:
                    found[behavior].append((group, recording, first, last))
    scores, start = Scores(), time.time()
    for behavior, segments in found.items():
        groups = {group : [s for s in segments if s[0] == group] for group in sorted({s[0] for s in segments})}
        chosen = [members[i] for members in groups.values() for i in rng.permutation(len(members))[:args.per_behavior]]
        for group, recording, first, last in chosen:
            end = min(last + args.after, len(recording) - 1)
            frames = np.arange(max(first, end - args.max_frames * args.stride), end + 1, args.stride)
            scene = recording.states[frames[0]].astype(np.float64)
            others = [b for b in GOALS if GOALS[b][0] != GOALS[behavior][0] and joint_side(b, scene[JOINT_OF[b][0]]) == 0]
            goals = [(behavior, args.max_conditions)] + [(str(b), args.max_conditions) for b in rng.choice(others, min(args.others, len(others)), replace = False)]
            query = torch.cat([embed(rendered(sim, recording.states, recording.proprio, frames[i:i + 16])) for i in range(0, len(frames), 16)])
            for goal, limit in goals:
                images = pairs_in_scene(sim, goal, poses[goal], scene, recording.proprio[last, :3] if goal == behavior else None, limit, rng)
                if len(images[0]) < 2:
                    print(f"{recording.name}:{first} {goal}: {len(images[0])} poses fit the scene, skipped")
                    continue
                score = change_score(embed(images[0]), embed(images[1]), query).cpu().numpy()
                scores.add(group + (" (its goal)" if goal == behavior else " (another goal)"), f"{recording.name}:{first}", goal, frames, score,
                           recording.states[frames, JOINT_OF[goal][0]], end)
            print(f"{behavior} {recording.name}:{first}: {len(frames)} frames, {[goal for goal, _ in goals]}, {time.time() - start:.0f} s", flush = True)
            scores.save(scores_file)
    analyze(scores, args, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest = "mode", required = True)
    common = argparse.ArgumentParser(add_help = False)
    common.add_argument("--guidance", default = "dynamics_model_5900.pth", help = "dynamics model checkpoint")
    common.add_argument("--device", default = DEVICE)
    common.add_argument("--env_config", default = DEFAULT_ENV_CONFIG)
    common.add_argument("--stride", type = int, default = 4, help = "every n-th env step. patience is converted")
    common.add_argument("--analyze_only", action = "store_true", help = "only evaluate the saved scores again")
    common.add_argument("--margins", type = float, nargs = "+", default = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    common.add_argument("--patience", type = int, nargs = "+", default = [5, 10, 20])
    common.add_argument("--tolerance", type = float, nargs = "+", default = [0.05])
    p = modes.add_parser("live", help = "what the switch did in finished rollouts")
    p.add_argument("--results", nargs = "+", required = True, help = "output folders of run_experiments.py")
    p.add_argument("--output", default = None, help = "default: the first results folder")
    p = modes.add_parser("replay", parents = [common], help = "the switch over the frames of finished rollouts")
    p.add_argument("--results", nargs = "+", required = True, help = "output folders of run_experiments.py")
    p.add_argument("--output", default = None, help = "default: the first results folder")
    p.add_argument("--task", default = None, help = "folder of the made task, default: the folder of each step's pos_examples")
    p.add_argument("--max_episodes", type = int, default = None, help = "per task")
    p = modes.add_parser("recordings", parents = [common], help = "the switch over any recordings")
    p.add_argument("--recordings", nargs = "+", required = True, help = "CALVIN folders, or rollout files of single-goal experiments")
    p.add_argument("--output", required = True)
    p.add_argument("--poses", default = os.path.join("datasets", "robot_poses", "articulated", "poses"), help = "the recorded poses of each behavior")
    p.add_argument("--per_behavior", type = int, default = 8, help = "recordings per behavior, for CALVIN data and rollouts each")
    p.add_argument("--others", type = int, default = 2, help = "goals that aren't the recorded behavior, per recording")
    p.add_argument("--max_conditions", type = int, default = 10)
    p.add_argument("--after", type = int, default = 45, help = "frames after the end of the recorded behavior")
    p.add_argument("--max_frames", type = int, default = 60)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    {"live" : live, "replay" : replay, "recordings" : recordings}[args.mode](args)
