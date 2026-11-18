import argparse
import json
import sys

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_behaviors import validate_experiment_config
from core.goal_sequence import is_multigoal, validate_multigoal_config
from rollout_tasks import GUIDANCE_SETTINGS, add_arguments, check_files, guidance_settings, list_tasks, prepare_output, run_tasks

"""
Runs a set of experiments unattended: each experiment config with each of its sets of guidance conditions, and the base policy
without guidance to compare. Starting the same command again continues where it stopped.

    python dynaguide_scripts/run_experiments.py --plan calvin_exp_configs_examples/experiments/plan_first_articulated_1.json \
        --output_folder results/experiments_first --agent base_policy_300.pth --guidance dynamics_model_5900.pth

    --dry_run                   only lists the tasks and checks the plan and all files
    --experiments drawer_open   only these experiments of the plan
    --status_minutes 2          how often a running task prints how far it is

The plan:
    n_rollouts, horizon         per task
    scale, ss, alpha            guidance settings, each experiment can override them
    base_policy                 true adds a task per experiment with --scale 0 --ss 1, the base policy without guidance
    keep_images                 true keeps the images in the rollout files, so create_guidance_conditions.py can use them
    experiments                 name : {"config" : experiment config, "conditions" : {variant : conditions}, "scale" : ..., ...}
                                Each variant replaces pos_examples of the config: a path, a list of paths, or patterns with *.
                                Without "conditions", the config's own pos_examples is the only variant, called "guided"
                                "seed" : n is optional, by default the experiment's position in the plan (0, 1, ...)
                                A multi-goal config (goal_sequence, see core/goal_sequence.py) has its conditions in its steps, so
                                instead of "conditions" it takes "switch_variants" : {variant : switch settings}, each merged into
                                the config's "switch", like {"latent" : {}, "oracle" : {"mode" : "oracle"}}

All tasks of one experiment get the same seed, so the variants and the base policy start from the same scenes and robot poses.
A plan split over several machines keeps its seeds by setting "seed" in each part.

The output folder:
    plan.json                   the plan of this run, a different plan can't continue in this folder
    summary.json                every task with its status and successes, updated after each task
    <experiment>/<variant>/     one task: a normal run_dynaguide.py output folder plus config.json, console.txt and DONE
    _incomplete/                tasks that crashed or were interrupted, moved here before they run again

Running the tasks is shared with harvest_rollouts.py, see rollout_tasks.py.
"""

EXPERIMENT_KEYS = ["config", "conditions", "switch_variants", "seed"] + GUIDANCE_SETTINGS
MULTIGOAL_TARGET = "SUCCESS" # what run_dynaguide.py prints on the Rollout line of a multi-goal episode that did every step
BASE_POLICY = "base_policy"


def make_tasks(plan):
    """Every (experiment, variant) of the plan, in a fixed order."""
    for key in ["experiments", "n_rollouts", "horizon"] + GUIDANCE_SETTINGS:
        if key not in plan:
            raise KeyError(f"The plan has no '{key}'")
    tasks = list()
    for number, (name, entry) in enumerate(plan["experiments"].items()):
        unknown = [key for key in entry if key not in EXPERIMENT_KEYS]
        if "config" not in entry or len(unknown) > 0:
            raise ValueError(f"Experiment {name}: needs 'config', can have {EXPERIMENT_KEYS[1:]}. Unknown keys: {unknown}")
        with open(entry["config"], "r") as f:
            config = json.load(f)
        guidance = guidance_settings(plan, entry)
        multigoal = is_multigoal(config)
        if multigoal:
            validate_multigoal_config(config)
            if "conditions" in entry:
                raise ValueError(f"Experiment {name}: a goal_sequence has its conditions in its steps, use switch_variants instead of conditions")
            targets = [MULTIGOAL_TARGET]
            variants = {variant : {**config, "switch" : {**config.get("switch", {}), **switch}}
                        for variant, switch in entry.get("switch_variants", {"guided" : {}}).items()}
        else:
            validate_experiment_config(config)
            if "switch_variants" in entry:
                raise ValueError(f"Experiment {name}: switch_variants are only for configs with a goal_sequence")
            targets = config.get("target_behaviors", [])
            variants = {variant : {**config, "pos_examples" : conditions}
                        for variant, conditions in entry.get("conditions", {"guided" : config["pos_examples"]}).items()}
        if BASE_POLICY in variants:
            raise ValueError(f"Experiment {name}: '{BASE_POLICY}' is the name of the run without guidance, call the variant differently")
        seed = entry.get("seed", number)
        for variant, variant_config in variants.items():
            tasks.append({"folder" : f"{name}/{variant}", "config" : variant_config, "seed" : seed,
                          "guidance" : guidance, "targets" : targets, "columns" : {"experiment" : name, "variant" : variant}})
        if plan.get("base_policy", False): # the guidance code still needs conditions to load, with scale 0 they have no effect
            if multigoal: # the scale of a goal step would override --scale 0. The switch doesn't matter without guidance, the oracle is cheapest
                config = {**config, "switch" : {**config.get("switch", {}), "mode" : "oracle"},
                          "goal_sequence" : [{**entry, "scale" : 0} for entry in config["goal_sequence"]]}
            tasks.append({"folder" : f"{name}/{BASE_POLICY}", "config" : config, "seed" : seed,
                          "guidance" : {"scale" : 0, "ss" : 1, "alpha" : guidance["alpha"]}, "targets" : targets,
                          "columns" : {"experiment" : name, "variant" : BASE_POLICY}})
    return tasks


def condition_specs(config):
    """Every pos_examples / neg_examples / contrast_examples of a config, also those of the goal steps of a goal_sequence."""
    if not is_multigoal(config):
        return [config["pos_examples"]]
    specs = list()
    for entry in config["goal_sequence"]:
        specs += [entry[key] for key in ["pos_examples", "neg_examples"] if entry.get(key) is not None]
        if entry.get("switch", {}).get("contrast_examples") is not None:
            specs.append(entry["switch"]["contrast_examples"])
    return specs


def main(args):
    with open(args.plan, "r") as f:
        plan = json.load(f)
    tasks = make_tasks(plan)
    if args.experiments is not None:
        unknown = [name for name in args.experiments if name not in plan["experiments"]]
        if len(unknown) > 0:
            sys.exit(f"Experiments {unknown} aren't in the plan, choose from {list(plan['experiments'])}")
    selected = [task for task in tasks if args.experiments is None or task["columns"]["experiment"] in args.experiments]
    check_files([task["config"]["reset_poses"] for task in selected if task["config"].get("reset_poses") is not None]
                + ([] if args.dry_run else [args.agent, args.guidance]),
                [spec for task in selected for spec in condition_specs(task["config"])])

    if args.dry_run:
        list_tasks(selected, plan)
        return

    prepare_output(args.output_folder, plan)
    run_tasks(tasks, selected, plan, args, keep_images = plan.get("keep_images", False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser, "calvin_exp_configs_examples/experiments")
    parser.add_argument("--experiments", type = str, nargs = "+", default = None, help = "only these experiments of the plan")
    args = parser.parse_args()
    if not args.dry_run and (args.agent is None or args.guidance is None):
        parser.error("--agent and --guidance are needed to run rollouts")
    main(args)
