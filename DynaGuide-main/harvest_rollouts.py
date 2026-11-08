import argparse
import glob
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from core.calvin_behaviors import ARTICULATED, validate_experiment_config
from core.calvin_utils import STATE_WORDS

"""
Runs guided rollouts for every goal of a plan from every start state, to get more guidance conditions out of them.
Meant to run unattended: each task is its own run_dynaguide.py process, a failed task doesn't stop the others, and starting
the same command again continues where it stopped.

    python harvest_rollouts.py --plan calvin_exp_configs_examples/harvest/articulated.json \
        --output_folder results/harvest_articulated --agent base_policy_300.pth --guidance dynamics_model_5900.pth

    --dry_run           only lists the tasks and checks the plan
    --goals drawer_open door_left
                        only these goals of the plan
    --conditions_only   only builds the guidance conditions from the finished tasks

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
"""

RUN_SCRIPT = "run_dynaguide.py"
JOINTS = ["sliding_door", "drawer", "switch", "green_light"] # in the order of ARTICULATED
SHORT_NAMES = {"sliding_door" : "door", "green_light" : "led"} # for the folder names
MAX_FAILURES_IN_A_ROW = 3 # more probably means a broken setup, like a wrong checkpoint path, so stop instead of failing every task


def goal_joint(goal):
    """Returns the joint a goal moves and the word the joint has to start at, like ("drawer", "closed") for drawer_open."""
    for joint, (idx, low, high, rising, falling) in zip(JOINTS, ARTICULATED):
        if goal in (rising, falling):
            words = [word for word, value in STATE_WORDS[joint].items() if (value < low if goal == rising else value > high)]
            return joint, words[0]
    raise ValueError(f"Goal {goal} isn't a goal of an articulated joint, choose from {[name for entry in ARTICULATED for name in entry[3:]]}")


def make_tasks(plan):
    """Every (goal, start state) of the plan, in a fixed order so the seeds stay the same when a harvest continues."""
    for key in ["joints", "goals", "n_rollouts", "horizon", "scale", "ss", "alpha", "reset_poses"]:
        if key not in plan:
            raise KeyError(f"The plan has no '{key}'")
    tasks = list()
    for goal, conditions in plan["goals"].items():
        joint, start_word = goal_joint(goal)
        others = [name for name in plan["joints"] if name != joint]
        for words in itertools.product(*[plan["joints"][name] for name in others]):
            start = dict(zip(others, words))
            start[joint] = start_word
            start = {name : start[name] for name in list(plan["joints"]) + [joint] if name in start} # the order of the plan
            config = {"target_behaviors" : [goal], "env_setup" : start, "use_neg" : False,
                      "pos_examples" : conditions, "reset_poses" : plan["reset_poses"]}
            validate_experiment_config(config)
            name = "__".join(f"{SHORT_NAMES.get(joint, joint)}_{word}" for joint, word in start.items())
            tasks.append({"goal" : goal, "start" : name, "folder" : os.path.join(goal, name), "config" : config, "seed" : len(tasks)})
    return tasks


def check_files(plan, args):
    paths = list(plan["goals"].values()) + [plan["reset_poses"]] + ([] if args.dry_run or args.conditions_only else [args.agent, args.guidance])
    missing = [path for path in paths if path is None or not os.path.exists(path)]
    if len(missing) > 0:
        sys.exit(f"Missing files: {missing}. Paths are relative to the folder you start this from")


def prepare_output(args, plan):
    """Keeps the plan in the output folder, so a harvest only continues with the plan it started with."""
    os.makedirs(args.output_folder, exist_ok = True)
    plan_path = os.path.join(args.output_folder, "plan.json")
    if os.path.exists(plan_path):
        with open(plan_path, "r") as f:
            if json.load(f) != plan:
                sys.exit(f"{args.output_folder} holds a harvest of a different plan, choose a new output folder")
    else:
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent = 4)


def set_aside(folder, args):
    """Moves an unfinished task out of the way, so its half-written files don't end up in the conditions."""
    target = os.path.join(args.output_folder, "_incomplete", folder.replace(os.sep, "__") + datetime.now().strftime("__%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.dirname(target), exist_ok = True)
    shutil.move(os.path.join(args.output_folder, folder), target)
    print(f"  moved the unfinished earlier attempt to {target}")


def run_task(task, plan, args):
    """Runs one task in its own process. Returns its DONE numbers, or None if it failed."""
    folder = os.path.join(args.output_folder, task["folder"])
    os.makedirs(folder)
    config_path = os.path.join(folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(task["config"], f, indent = 4)

    command = [sys.executable, RUN_SCRIPT, "--agent", args.agent, "--guidance", args.guidance, "--exp_setup_config", config_path,
               "--output_folder", folder, "--n_rollouts", str(plan["n_rollouts"]), "--horizon", str(plan["horizon"]),
               "--scale", str(plan["scale"]), "--ss", str(plan["ss"]), "--alpha", str(plan["alpha"]), "--seed", str(args.seed + task["seed"]),
               "--camera_names", "third_person", "--video_skip", "2", "--no_collage", "--no_live_render", "--dataset_images"]
    start = time.time()
    with open(os.path.join(folder, "console.txt"), "w") as console:
        console.write(" ".join(command) + "\n\n")
        console.flush()
        returncode = subprocess.call(command, stdout = console, stderr = subprocess.STDOUT)

    logs, rollouts = glob.glob(os.path.join(folder, "*_log.json")), glob.glob(os.path.join(folder, "*_rollout.hdf5"))
    if returncode != 0 or len(logs) != 1 or len(rollouts) != 1: # run_dynaguide.py writes the log and closes the hdf5 only at the very end
        print(f"  FAILED with exit code {returncode}, see {os.path.join(folder, 'console.txt')}")
        return None
    with open(logs[0], "r") as f:
        log = json.load(f)
    done = {"successes" : log["Num_Success"], "rollouts" : log["Num_Episode"], "behaviors" : log["Behavior_Counts"],
            "minutes" : round((time.time() - start) / 60, 1), "rollout_file" : os.path.basename(rollouts[0])}
    with open(os.path.join(folder, "DONE"), "w") as f:
        json.dump(done, f, indent = 4)
    return done


def read_done(task, args):
    path = os.path.join(args.output_folder, task["folder"], "DONE")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def write_summary(tasks, statuses, args):
    rows = list()
    for task in tasks:
        done = read_done(task, args)
        rows.append({"goal" : task["goal"], "start" : task["start"], "status" : statuses.get(task["folder"], "done" if done else "open"),
                     "successes" : None if done is None else done["successes"], "rollouts" : None if done is None else done["rollouts"],
                     "minutes" : None if done is None else done["minutes"]})
    with open(os.path.join(args.output_folder, "summary.json"), "w") as f:
        json.dump(rows, f, indent = 4)
    return rows


def print_table(rows):
    print(f"\n{'goal':<14} {'start state':<44} {'status':<9} successes")
    for row in rows:
        result = "" if row["successes"] is None else f"{row['successes']} / {row['rollouts']}"
        print(f"{row['goal']:<14} {row['start']:<44} {row['status']:<9} {result}")


def build_conditions(tasks, args):
    """Cuts all finished rollouts into one condition file per behavior. The labels come from the states, not from the goal of the task."""
    from core.guidance_conditions import build_behavior_conditions, read_recordings # only needed here, and it pulls in more packages
    recordings = list()
    for task in tasks:
        done = read_done(task, args)
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
    check_files(plan, args)

    if args.dry_run:
        print(f"{len(selected)} tasks with {plan['n_rollouts']} rollouts each:")
        for task in selected:
            print(f"  {task['folder']}")
        return

    prepare_output(args, plan)
    if args.conditions_only:
        build_conditions(tasks, args)
        return

    statuses, failures_in_a_row, minutes = dict(), 0, list()
    todo = [task for task in selected if read_done(task, args) is None]
    print(f"{len(selected) - len(todo)} of {len(selected)} tasks are already done, {len(todo)} to go")
    for number, task in enumerate(todo):
        if os.path.exists(os.path.join(args.output_folder, task["folder"])):
            set_aside(task["folder"], args)
        eta = "" if len(minutes) == 0 else f", about {sum(minutes) / len(minutes) * (len(todo) - number) / 60:.1f} h left"
        print(f"[{datetime.now():%H:%M}] task {number + 1} / {len(todo)}: {task['folder']}{eta}", flush = True)

        try:
            done = run_task(task, plan, args)
        except KeyboardInterrupt:
            statuses[task["folder"]] = "stopped"
            write_summary(tasks, statuses, args)
            sys.exit("\nStopped. Start the same command again to continue, this task will run again")

        if done is None:
            statuses[task["folder"]] = "failed"
            failures_in_a_row += 1
            if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                write_summary(tasks, statuses, args)
                sys.exit(f"{failures_in_a_row} tasks failed in a row, stopping. Check console.txt of the last task")
        else:
            failures_in_a_row = 0
            minutes.append(done["minutes"])
            print(f"  {done['successes']} / {done['rollouts']} reached {task['goal']} in {done['minutes']} min, behaviors {done['behaviors']}", flush = True)
        write_summary(tasks, statuses, args)

    print_table(write_summary(tasks, statuses, args))
    build_conditions(tasks, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type = str, required = True, help = "the plan file, see calvin_exp_configs_examples/harvest")
    parser.add_argument("--output_folder", type = str, required = True, help = "new folder, or the folder of a harvest to continue")
    parser.add_argument("--agent", type = str, default = None, help = "base policy checkpoint")
    parser.add_argument("--guidance", type = str, default = None, help = "dynamics model checkpoint")
    parser.add_argument("--goals", type = str, nargs = "+", default = None, help = "only these goals of the plan")
    parser.add_argument("--seed", type = int, default = 0, help = "task i runs with seed + i")
    parser.add_argument("--dry_run", action = "store_true", help = "only list the tasks")
    parser.add_argument("--conditions_only", action = "store_true", help = "only build the conditions from the finished tasks")
    args = parser.parse_args()
    if not (args.dry_run or args.conditions_only) and (args.agent is None or args.guidance is None):
        parser.error("--agent and --guidance are needed to run rollouts")
    main(args)
