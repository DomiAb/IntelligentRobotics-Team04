import argparse
import glob
import json
import os
import random

import h5py
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import validate_experiment_config

"""
The spawn test: does a goal step later in a chain fail because of where the arm is when the step starts?

From recorded chain rollouts it takes the moment DynaGuide switched to a step (scene and arm exactly as they were) and writes two
reset files with the same scenes, paired episode by episode (run_dynaguide.py, reset_poses with scene_states):
    <step>_chain_start.json    the arm where the chain left it
    <step>_spawn.json          the arm at a spawn pose (initial_calvin_robot_states_right_side_midpoint.json, drawn with a fixed seed)
and a single-goal experiment config for each, with the step's conditions from the chain's task, and a plan per step. If spawn clearly
beats chain_start, the arm pose is the cause and a "go home" phase between the steps should help.

    python dynaguide_scripts/make_spawn_test.py --results results/goal_switch_ results/goal_switch_patches

Only episodes where the env had done all steps before (so the scene is the chain's), and the step itself wasn't done yet, are used.
Output: calvin_exp_configs_examples/spawn_test/
"""

CHAIN = "drawer_button_door"
STEPS = {"button_on" : {"index" : 1, "scale" : 1, "alpha" : 30, "horizon" : 250},
         "door_right" : {"index" : 2, "scale" : 2, "alpha" : 15, "horizon" : 400}}
ENV_WORDS = {"button_on" : {"drawer" : "open", "sliding_door" : "left", "switch" : "off", "green_light" : "off"},
             "door_right" : {"drawer" : "open", "sliding_door" : "left", "switch" : "off", "green_light" : "on"}} # the chain's scene, for the record
SPAWN_POSES = "calvin_exp_configs_examples/reset_poses/initial_calvin_robot_states_right_side_midpoint.json"
OUTPUT = "calvin_exp_configs_examples/spawn_test"


def scene_fits(step, state):
    """The scene the chain leaves for the step: drawer open, door left; the light off before button_on, on before door_right."""
    light = state[5] > 0.5
    return state[1] > 0.1 and state[0] > 0.25 and (not light if step == "button_on" else light)


def collect(results):
    starts = {step : list() for step in STEPS}
    logs = sorted(glob.glob(os.path.join(results, "**", CHAIN, "*", "*_log.json"), recursive = True))
    for log_file in logs:
        variant = os.path.basename(os.path.dirname(log_file))
        if variant in ["previous", "base_policy"]:
            continue
        with open(log_file) as f:
            log = json.load(f)
        with h5py.File(log_file.replace("_log.json", "_rollout.hdf5"), "r") as f:
            demos = sorted(f["data"].keys(), key = lambda name: int(name.split("_")[-1]))
            for episode, demo in enumerate(demos):
                goal_steps, switches = log["Episode_Goal_Steps"][episode]["steps"], log["Episode_Switches"][episode]["switches"]
                for step, info in STEPS.items():
                    k = info["index"]
                    t = next((s["t"] for s in switches if s["from_step"] == k - 1), None) # DynaGuide starts guiding to the step
                    if t is None or k >= len(goal_steps) + 1:
                        continue
                    done_before = all(goal_steps[j]["completed_at"] is not None and goal_steps[j]["completed_at"] <= t for j in range(k))
                    own = goal_steps[k]["completed_at"] if k < len(goal_steps) else None
                    if not done_before or (own is not None and own <= t):
                        continue
                    states, proprio = f["data"][demo]["obs"]["states"], f["data"][demo]["obs"]["proprio"]
                    if t >= len(states):
                        continue
                    state, robot = states[t].astype(np.float64), proprio[t].astype(np.float64)
                    if not scene_fits(step, state):
                        continue
                    starts[step].append({"scene" : state.tolist(), "robot" : robot.tolist(),
                                         "source" : f"{os.path.relpath(log_file)}:{demo}:t{t}"})
    return starts


def main(args):
    starts = {step : list() for step in STEPS}
    for results in args.results:
        for step, found in collect(results).items():
            starts[step] += found
    with open(SPAWN_POSES) as f:
        spawn = json.load(f)["robot_states"]
    os.makedirs(args.output, exist_ok = True)
    rng = random.Random(0)
    for step, found in starts.items():
        info = STEPS[step]
        if len(found) == 0:
            raise SystemExit(f"No start for {step} in {args.results}")
        scenes = [entry["scene"] for entry in found]
        variants = {"chain_start" : [entry["robot"] for entry in found], "spawn" : [rng.choice(spawn) for _ in found]}
        plan = {"description" : f"The spawn test for {step} of {CHAIN}: the same {len(found)} scenes where DynaGuide switched to the step, "
                                "the arm where the chain left it (chain_start) or at a spawn pose (spawn). One rollout per scene. "
                                "Made with dynaguide_scripts/make_spawn_test.py, run with dynaguide_scripts/run_spawn_test.bat",
                "n_rollouts" : len(found), "horizon" : info["horizon"], "scale" : info["scale"], "ss" : 4, "alpha" : info["alpha"],
                "base_policy" : False, "keep_images" : False, "video" : False, "experiments" : dict()}
        for variant, robots in variants.items():
            reset_file = os.path.join(args.output, f"{step}_{variant}.json")
            with open(reset_file, "w") as f:
                json.dump({"robot_states" : robots, "scene_states" : scenes, "sources" : [entry["source"] for entry in found]}, f)
            config = {"description" : f"{step} alone, starting from the scenes of {CHAIN} where DynaGuide switched to it, arm: {variant}",
                      "target_behaviors" : [step], "env_setup" : ENV_WORDS[step], "use_neg" : False,
                      "pos_examples" : f"datasets/guidance_conditions/{CHAIN}/{step}.hdf5", "reset_poses" : reset_file.replace(os.sep, "/")}
            validate_experiment_config(config)
            config_file = os.path.join(args.output, f"{step}_{variant}_config.json")
            with open(config_file, "w") as f:
                json.dump(config, f, indent = 4)
            plan["experiments"][f"{step}_{variant}"] = {"config" : config_file.replace(os.sep, "/"), "seed" : 0}
        with open(os.path.join(args.output, f"plan_{step}.json"), "w") as f:
            json.dump(plan, f, indent = 4)
        distance = np.median([np.linalg.norm(np.array(a[:3]) - np.array(b[:3])) for a, b in zip(variants["chain_start"], variants["spawn"])])
        print(f"{step}: {len(found)} starts, the chain's arm is {distance:.2f} m (median) from the spawn pose")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs = "+", required = True, help = "folders with recorded runs of the drawer_button_door chain")
    parser.add_argument("--output", default = OUTPUT)
    main(parser.parse_args())
