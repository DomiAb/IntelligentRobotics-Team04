"""
Use this script to score a saved CALVIN rollout file against the goals of an experiment config, without running the rollouts again.
python evaluate_behaviors.py \
    --dataset results/my_run \
    --config calvin_exp_configs_examples/switch_on.json \
    --output results/behaviors.json
"""

import argparse
import glob
import json
import os

from core.calvin_behaviors import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description="Label what the robot did in each rollout and calculate the success rate.")
    parser.add_argument("--dataset", required=True, help="Output folder of a run, or a rollout file that was saved with --dataset_obs")
    parser.add_argument("--config", required=True, help="Experiment config with target_behaviors and/or avoid_behaviors")
    parser.add_argument("--output", default=None, help="Optional path to save the results as json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    dataset = args.dataset
    if os.path.isdir(dataset):
        rollout_files = glob.glob(os.path.join(dataset, "*.hdf5"))
        if len(rollout_files) != 1:
            raise FileNotFoundError(f"Expected one hdf5 file in {dataset}, found {rollout_files}. Pass the file instead.")
        dataset = rollout_files[0]

    results = evaluate_dataset(dataset, config)
    results_json = json.dumps(results, indent=4)
    print(results_json)
    if args.output is not None:
        if os.path.dirname(args.output) != "":
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(results_json)


if __name__ == "__main__":
    main()
