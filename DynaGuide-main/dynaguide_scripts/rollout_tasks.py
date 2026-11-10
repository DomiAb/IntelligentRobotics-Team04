import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_utils import condition_files

"""
What harvest_rollouts.py and run_experiments.py share: running a list of tasks unattended, each task one run_dynaguide.py process.
A failed task doesn't stop the others, a running task prints a status line now and then, and starting the same command again
continues where it stopped.

A task is a dict:
    folder      where it goes inside the output folder, like "drawer_open/blocks"
    config      its experiment config, written to <folder>/config.json
    seed        added to --seed
    guidance    {"scale", "ss", "alpha"}
    targets     the behaviors that count as reached in the status lines
    columns     what the summary shows about it, like {"experiment" : "drawer_open", "variant" : "blocks"}

Each task folder is a normal run_dynaguide.py output folder plus config.json, console.txt (everything run_dynaguide.py printed) and
DONE (written once the task finished, with its numbers). Unfinished tasks are moved to _incomplete/ before they run again.
"""

RUN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_dynaguide.py") # next to this script
MAX_FAILURES_IN_A_ROW = 3 # more probably means a broken setup, like a wrong checkpoint path, so stop instead of failing every task
GUIDANCE_SETTINGS = ["scale", "ss", "alpha"] # set for the whole plan, and each entry of the plan can override them
QUIET_WARNING_MINUTES = 10 # run_dynaguide.py prints every 14 steps, so a longer silence means it is probably stuck


def add_arguments(parser, plan_folder):
    """The command line options both scripts have."""
    parser.add_argument("--plan", type = str, required = True, help = f"the plan file, see {plan_folder}")
    parser.add_argument("--output_folder", type = str, required = True, help = "new folder, or the folder of a run to continue")
    parser.add_argument("--agent", type = str, default = None, help = "base policy checkpoint")
    parser.add_argument("--guidance", type = str, default = None, help = "dynamics model checkpoint")
    parser.add_argument("--seed", type = int, default = 0, help = "added to the seed of each task")
    parser.add_argument("--status_minutes", type = float, default = 2, help = "minutes between two status lines while a task runs")
    parser.add_argument("--dry_run", action = "store_true", help = "only list the tasks and check the plan")


def guidance_settings(plan, entry):
    """scale, ss and alpha of the plan, with what this entry of the plan overrides."""
    return {key : entry.get(key, plan[key]) for key in GUIDANCE_SETTINGS}


def check_files(paths, condition_specs):
    """Stops if a file is missing: plain paths, and guidance conditions as in pos_examples (a path, a list, patterns with *)."""
    missing = [path for path in paths if path is None or not os.path.exists(path)]
    for spec in condition_specs:
        try:
            condition_files(spec)
        except (FileNotFoundError, ValueError) as error:
            missing.append(str(error))
    if len(missing) > 0:
        sys.exit("Missing files:\n  " + "\n  ".join(missing) + "\nPaths are relative to the folder you start this from")


def prepare_output(output_folder, plan):
    """Keeps the plan in the output folder, so a run only continues with the plan it started with."""
    os.makedirs(output_folder, exist_ok = True)
    plan_path = os.path.join(output_folder, "plan.json")
    if os.path.exists(plan_path):
        with open(plan_path, "r") as f:
            if json.load(f) != plan:
                sys.exit(f"{output_folder} holds a run of a different plan, choose a new output folder")
    else:
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent = 4)


def set_aside(folder, output_folder):
    """Moves an unfinished task out of the way, so its half-written files don't count."""
    target = os.path.join(output_folder, "_incomplete", folder.replace(os.sep, "__").replace("/", "__") + datetime.now().strftime("__%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.dirname(target), exist_ok = True)
    shutil.move(os.path.join(output_folder, folder), target)
    print(f"  moved the unfinished earlier attempt to {target}")


def progress(console_path, task, plan):
    """Reads how far run_dynaguide.py got from what it printed: finished rollouts, and steps of the current one."""
    with open(console_path, "r", errors = "replace") as f:
        lines = f.read().splitlines()
    finished = [line for line in lines if line.startswith("Rollout ") and " behavior: " in line]
    reached = sum(any(target in line for target in task["targets"]) for line in finished)
    last_finished = max([i for i, line in enumerate(lines) if line in finished] + [-1])
    steps = 14 * sum(line.startswith("Guidance norms") for line in lines[last_finished + 1:]) # printed every 14 steps
    if len(finished) == 0 and steps == 0:
        return "loading the models and the conditions"
    return f"rollout {len(finished) + 1} / {plan['n_rollouts']} at step ~{steps} / {plan['horizon']}, goal reached {reached} / {len(finished)} so far"


def watch(process, console_path, task, plan, status_minutes):
    """Waits for the task, and prints a status line every status_minutes so you can tell it is still running."""
    start = last_status = last_output = time.time()
    last_size = 0
    while True:
        try:
            return process.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        size = os.path.getsize(console_path)
        if size != last_size:
            last_size, last_output = size, now
        if now - last_status >= status_minutes * 60:
            line = f"  [{datetime.now():%H:%M}] running for {(now - start) / 60:.0f} min: {progress(console_path, task, plan)}"
            if now - last_output >= QUIET_WARNING_MINUTES * 60:
                line += f". WARNING: no new output for {(now - last_output) / 60:.0f} min, look at {console_path}"
            print(line, flush = True)
            last_status = now


def run_task(task, plan, args, keep_images):
    """Runs one task in its own process. Returns its DONE numbers, or None if it failed."""
    folder = os.path.join(args.output_folder, task["folder"])
    os.makedirs(folder)
    config_path = os.path.join(folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(task["config"], f, indent = 4)

    command = [sys.executable, RUN_SCRIPT, "--agent", args.agent, "--guidance", args.guidance, "--exp_setup_config", config_path,
               "--output_folder", folder, "--n_rollouts", str(plan["n_rollouts"]), "--horizon", str(plan["horizon"]),
               "--seed", str(args.seed + task["seed"]), "--camera_names", "third_person", "--video_skip", "2",
               "--no_collage", "--no_live_render"] + (["--dataset_images"] if keep_images else [])
    for key in GUIDANCE_SETTINGS:
        command += ["--" + key, str(task["guidance"][key])]
    start = time.time()
    console_path = os.path.join(folder, "console.txt")
    with open(console_path, "w") as console:
        console.write(" ".join(command) + "\n\n")
        console.flush()
        process = subprocess.Popen(command, stdout = console, stderr = subprocess.STDOUT)
        try:
            returncode = watch(process, console_path, task, plan, args.status_minutes)
        except KeyboardInterrupt:
            process.kill()
            raise

    logs, rollouts = glob.glob(os.path.join(folder, "*_log.json")), glob.glob(os.path.join(folder, "*_rollout.hdf5"))
    if returncode != 0 or len(logs) != 1 or len(rollouts) != 1: # run_dynaguide.py writes the log and closes the hdf5 only at the very end
        print(f"  FAILED with exit code {returncode}, see {console_path}")
        return None
    with open(logs[0], "r") as f:
        log = json.load(f)
    done = {"successes" : log["Num_Success"], "rollouts" : log["Num_Episode"], "behaviors" : log["Behavior_Counts"],
            "minutes" : round((time.time() - start) / 60, 1), "rollout_file" : os.path.basename(rollouts[0])}
    with open(os.path.join(folder, "DONE"), "w") as f:
        json.dump(done, f, indent = 4)
    return done


def read_done(task, output_folder):
    path = os.path.join(output_folder, task["folder"], "DONE")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def write_summary(tasks, statuses, output_folder):
    rows = list()
    for task in tasks:
        done = read_done(task, output_folder)
        rows.append({**task["columns"], "status" : statuses.get(task["folder"], "done" if done else "open"),
                     **{key : None if done is None else done[key] for key in ["successes", "rollouts", "minutes"]}})
    with open(os.path.join(output_folder, "summary.json"), "w") as f:
        json.dump(rows, f, indent = 4)
    return rows


def print_table(rows):
    columns = [key for key in rows[0] if key not in ["status", "successes", "rollouts", "minutes"]]
    widths = {key : max(len(key), max(len(str(row[key])) for row in rows)) + 2 for key in columns}
    print("\n" + "".join(f"{key:<{widths[key]}}" for key in columns) + f"{'status':<9} successes")
    for row in rows:
        result = "" if row["successes"] is None else f"{row['successes']} / {row['rollouts']}"
        print("".join(f"{str(row[key]):<{widths[key]}}" for key in columns) + f"{row['status']:<9} {result}")


def list_tasks(selected, plan):
    print(f"{len(selected)} tasks with {plan['n_rollouts']} rollouts each:")
    for task in selected:
        settings = ", ".join(f"{key} {value}" for key, value in task["guidance"].items())
        print(f"  {task['folder']:<50} {settings}, seed +{task['seed']}")


def run_tasks(tasks, selected, plan, args, keep_images):
    """Runs the selected tasks that aren't done yet, one after the other. Returns the summary rows of all tasks."""
    statuses, failures_in_a_row, minutes = dict(), 0, list()
    todo = [task for task in selected if read_done(task, args.output_folder) is None]
    print(f"{len(selected) - len(todo)} of {len(selected)} tasks are already done, {len(todo)} to go")
    for number, task in enumerate(todo):
        if os.path.exists(os.path.join(args.output_folder, task["folder"])):
            set_aside(task["folder"], args.output_folder)
        eta = "" if len(minutes) == 0 else f", about {sum(minutes) / len(minutes) * (len(todo) - number) / 60:.1f} h left"
        print(f"[{datetime.now():%H:%M}] task {number + 1} / {len(todo)}: {task['folder']}{eta}", flush = True)

        try:
            done = run_task(task, plan, args, keep_images)
        except KeyboardInterrupt:
            statuses[task["folder"]] = "stopped"
            write_summary(tasks, statuses, args.output_folder)
            sys.exit("\nStopped. Start the same command again to continue, this task will run again")

        if done is None:
            statuses[task["folder"]] = "failed"
            failures_in_a_row += 1
            if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                write_summary(tasks, statuses, args.output_folder)
                sys.exit(f"{failures_in_a_row} tasks failed in a row, stopping. Check console.txt of the last task")
        else:
            failures_in_a_row = 0
            minutes.append(done["minutes"])
            print(f"  {done['successes']} / {done['rollouts']} reached {' or '.join(task['targets'])} in {done['minutes']} min, "
                  f"behaviors {done['behaviors']}", flush = True)
        write_summary(tasks, statuses, args.output_folder)

    rows = write_summary(tasks, statuses, args.output_folder)
    print_table(rows)
    return rows
