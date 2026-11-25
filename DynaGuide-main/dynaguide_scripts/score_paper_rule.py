import argparse
import glob
import json
import os

import h5py
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import ARTICULATED, COMPLETED, episode_states, label_first_interaction

"""
Scores the finished tasks of a run_experiments.py folder with three rules, for experiments with one articulated target behavior.
No rollouts, it reads the saved ones.

    python dynaguide_scripts/score_paper_rule.py --results results/doors_paper

    authors      the plotting code of the paper (analyze_calvin_touch.py): the first thing that happens in the episode is the target.
                 5 cm of the door or the drawer is enough, and touching a block first counts as a failure
    halfway      the joint went past half of its range at any time, what the text of the paper says
    finished     the joint reached COMPLETED (core/calvin_behaviors.py) at any time, our rule

Prints the table and writes it to <results>/paper_rule.json. (95% intervals: Wilson)
"""

RANGES = {0 : 0.27, 1 : 0.16, 3 : 0.09, 5 : 1.0} # of the joints, as in check_state_difference
JOINT = {name : (idx, sign) for idx, _, _, rising, falling in ARTICULATED for name, sign in [(rising, 1), (falling, -1)]}


def wilson(successes, n, z = 1.96):
    p = successes / n
    center, width = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (center - width) / (1 + z * z / n)), 3), round((center + width) / (1 + z * z / n), 3)]


def score_episode(states, proprio, target):
    idx, sign = JOINT[target]
    joint = states[:, idx]
    middle = RANGES[idx] / 2
    end = COMPLETED.get(target, 0.8 if sign > 0 else 0.2)
    label = label_first_interaction(states, proprio)
    return {"authors" : "sanity_" + target in label,
            "halfway" : bool(joint.max() > middle if sign > 0 else joint.min() < middle),
            "finished" : bool(joint.max() > end if sign > 0 else joint.min() < end),
            "first" : "block" if any(name.endswith("_displace") for name in label) else label[0][len("sanity_"):]}


def main(args):
    rows = list()
    for done in sorted(glob.glob(os.path.join(args.results, "*", "*", "DONE"))):
        folder = os.path.dirname(done)
        with open(os.path.join(folder, "config.json"), "r") as f:
            targets = json.load(f).get("target_behaviors", [])
        files = sorted(glob.glob(os.path.join(folder, "*_rollout.hdf5")))
        if len(targets) != 1 or targets[0] not in JOINT or len(files) == 0:
            print(f"{folder}: skipped, it needs one articulated target behavior and a rollout file")
            continue
        with h5py.File(files[-1], "r") as file:
            episodes = list()
            for demo in sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1])):
                states, proprio = episode_states(file["data"][demo])
                if states.ndim == 3: # the policy stacks frames
                    states, proprio = states[:, -1], proprio[:, -1]
                episodes.append(score_episode(states, proprio, targets[0]))
        n = len(episodes)
        row = {"task" : os.path.relpath(folder, args.results).replace(os.sep, "/"), "n" : n}
        for rule in ["authors", "halfway", "finished"]:
            count = sum(episode[rule] for episode in episodes)
            row[rule] = {"successes" : count, "rate" : round(count / n, 3), "interval" : wilson(count, n)}
        names = [episode["first"] for episode in episodes]
        row["first_interaction"] = {name : names.count(name) for name in sorted(set(names), key = names.count, reverse = True)}
        rows.append(row)

    print(f"{'task':26s} {'n':>3s}   {'authors':>18s} {'halfway':>18s} {'finished':>18s}   what happened first")
    for row in rows:
        cells = [f"{row[rule]['successes']:3d} {100 * row[rule]['rate']:3.0f}% ({100 * row[rule]['interval'][0]:.0f}-{100 * row[rule]['interval'][1]:.0f})"
                 for rule in ["authors", "halfway", "finished"]]
        print(f"{row['task']:26s} {row['n']:3d}   " + " ".join(f"{cell:>18s}" for cell in cells) + "   "
              + ", ".join(f"{name} {count}" for name, count in row["first_interaction"].items()))
    with open(os.path.join(args.results, "paper_rule.json"), "w") as f:
        json.dump(rows, f, indent = 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type = str, required = True, help = "output folder of run_experiments.py")
    main(parser.parse_args())
