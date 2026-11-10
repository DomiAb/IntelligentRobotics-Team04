import argparse
import itertools
import json
import os
import sys
from datetime import datetime

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import ARTICULATED, validate_experiment_config
from core.calvin_utils import STATE_WORDS
from rollout_tasks import GUIDANCE_SETTINGS, add_arguments, check_files, guidance_settings, list_tasks, prepare_output, read_done, run_tasks

"""
Runs guided rollouts for every goal of a plan from every start state, to get more guidance conditions out of them.
Meant to run unattended: each task is its own run_dynaguide.py process, a failed task doesn't stop the others, and starting
the same command again continues where it stopped.

    python dynaguide_scripts/harvest_rollouts.py --plan calvin_exp_configs_examples/harvest/articulated.json \
        --output_folder results/harvest_articulated --agent base_policy_300.pth --guidance dynamics_model_5900.pth

    --dry_run           only lists the tasks and checks the plan
    --goals drawer_open door_left
                        only these goals of the plan
    --conditions_only   only builds the guidance conditions from the finished tasks
    --status_minutes 2  how often a running task prints how far it is, by default every 2 minutes

A goal of the plan is its guidance conditions (a path, a list of paths, or patterns with *, like pos_examples), or
{"conditions" : ..., "scale" : 1, "alpha" : 40} to change the guidance settings of the plan for this goal. There is no PyBullet window: --no_live_render drops it, the images stay the same.

A task is one goal from one start state, like drawer_open from drawer_closed__door_left__switch_on. The goal's own joint
starts where the goal can happen (closed for drawer_open), the other joints of the plan take every combination of their words.
Everything not in the plan stays random: the blocks, the green light, and the robot start pose out of reset_poses.

The output folder:
    plan.json                       the plan of this harvest, a different plan can't continue in this folder
    summary.json                    every task with its status and successes, updated after each task
    <goal>/<start state>/           one task, a normal run_dynaguide.py output folder plus
        config.json                 its experiment config, also works with preview_resets.py
        console.txt                 everything run_dynaguide.py printed
        DONE                        written once the task finished, with its numbers
    _incomplete/                    tasks that crashed or were interrupted, moved here before they run again
    conditions_<time>/              guidance conditions cut from all finished rollouts, see "Create guidance conditions.ipynb"

Running the tasks is shared with run_experiments.py, see rollout_tasks.py.
"""

JOINTS = ["sliding_door", "drawer", "switch", "green_light"] # in the order of ARTICULATED
SHORT_NAMES = {"sliding_door" : "door", "green_light" : "led"} # for the folder names


def goal_joint(goal):
    """Returns the joint a goal moves and the word the joint has to start at, like ("drawer", "closed") for drawer_open."""
    for joint, (idx, low, high, rising, falling) in zip(JOINTS, ARTICULATED):
        if goal in (rising, falling):
            words = [word for word, value in STATE_WORDS[joint].items() if (value < low if goal == rising else value > high)]
            return joint, words[0]
    raise ValueError(f"Goal {goal} isn't a goal of an articulated joint, choose from {[name for entry in ARTICULATED for name in entry[3:]]}")


def goal_entry(plan, goal):
    """A goal of the plan as a dict: its conditions, and maybe its own scale, ss and alpha."""
    entry = plan["goals"][goal]
    entry = {"conditions" : entry} if isinstance(entry, (str, list)) else entry
    unknown = [key for key in entry if key not in ["conditions"] + GUIDANCE_SETTINGS]
    if "conditions" not in entry or len(unknown) > 0:
        raise ValueError(f"Goal {goal} of the plan: give its conditions, or a dict with 'conditions' and optionally {GUIDANCE_SETTINGS}. Unknown keys: {unknown}")
    return entry


def make_tasks(plan):
    """Every (goal, start state) of the plan, in a fixed order so the seeds stay the same when a harvest continues."""
    for key in ["joints", "goals", "n_rollouts", "horizon", "reset_poses"] + GUIDANCE_SETTINGS:
        if key not in plan:
            raise KeyError(f"The plan has no '{key}'")
    tasks = list()
    for goal in plan["goals"]:
        entry = goal_entry(plan, goal)
        joint, start_word = goal_joint(goal)
        others = [name for name in plan["joints"] if name != joint]
        for words in itertools.product(*[plan["joints"][name] for name in others]):
            start = dict(zip(others, words))
            start[joint] = start_word
            start = {name : start[name] for name in list(plan["joints"]) + [joint] if name in start} # the order of the plan
            config = {"target_behaviors" : [goal], "env_setup" : start, "use_neg" : False,
                      "pos_examples" : entry["conditions"], "reset_poses" : plan["reset_poses"]}
            validate_experiment_config(config)
            name = "__".join(f"{SHORT_NAMES.get(joint, joint)}_{word}" for joint, word in start.items())
            tasks.append({"folder" : os.path.join(goal, name), "config" : config, "seed" : len(tasks), "guidance" : guidance_settings(plan, entry),
                          "targets" : [goal], "columns" : {"goal" : goal, "start" : name}, "goal" : goal, "start" : name})
    return tasks


def build_conditions(tasks, args):
    """Cuts all finished rollouts into one condition file per behavior. The labels come from the states, not from the goal of the task."""
    from core.guidance_conditions import build_behavior_conditions, read_recordings # only needed here, and it pulls in more packages
    recordings = list()
    for task in tasks:
        done = read_done(task, args.output_folder)
        if done is None:
            continue
        for recording in read_recordings([os.path.join(args.output_folder, task["folder"], done["rollout_file"])]):
            start = task["start"].replace("__", "/") # the preview puts each part between / on its own line
            recording.name = f"{task['goal']}/{start}/demo_{recording.name.rsplit('_', 1)[1]}" # instead of the date of the file
            recordings.append(recording)
    if len(recordings) == 0:
        print("No finished tasks yet, so no conditions")
        return
    folder = os.path.join(args.output_folder, "conditions_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    counts = build_behavior_conditions(recordings, folder)
    print(f"\nConditions from {len(recordings)} rollouts in {folder}:")
    for name, count in counts.items():
        print(f"  {name}: {count}")


def main(args):
    with open(args.plan, "r") as f:
        plan = json.load(f)
    tasks = make_tasks(plan)
    if args.goals is not None:
        unknown = [goal for goal in args.goals if goal not in plan["goals"]]
        if len(unknown) > 0:
            sys.exit(f"Goals {unknown} aren't in the plan, choose from {list(plan['goals'])}")
    selected = [task for task in tasks if args.goals is None or task["goal"] in args.goals]
    check_files([plan["reset_poses"]] + ([] if args.dry_run or args.conditions_only else [args.agent, args.guidance]),
                [goal_entry(plan, goal)["conditions"] for goal in plan["goals"]])

    if args.dry_run:
        list_tasks(selected, plan)
        return

    prepare_output(args.output_folder, plan)
    if not args.conditions_only:
        run_tasks(tasks, selected, plan, args, keep_images = True) # the images are what the conditions are made of
    build_conditions(tasks, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser, "calvin_exp_configs_examples/harvest")
    parser.add_argument("--goals", type = str, nargs = "+", default = None, help = "only these goals of the plan")
    parser.add_argument("--conditions_only", action = "store_true", help = "only build the conditions from the finished tasks")
    args = parser.parse_args()
    if not (args.dry_run or args.conditions_only) and (args.agent is None or args.guidance is None):
        parser.error("--agent and --guidance are needed to run rollouts")
    main(args)
