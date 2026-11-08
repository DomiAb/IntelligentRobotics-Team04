import argparse
import json
import os
import sys

from core.guidance_conditions import build_behavior_conditions, build_goal_conditions, read_recordings

"""
Creates guidance conditions from recorded data, see "Create guidance conditions.ipynb" for how to use it.
Each condition file works as pos_examples or neg_examples in an experiment config.

One file per behavior, cut from the recordings:
    python create_guidance_conditions.py --recordings datasets/calvin_debug_dataset/validation --output_folder datasets/guidance_conditions/debug_validation

One file per goal of a goals file, found by a description of the state:
    python create_guidance_conditions.py --recordings datasets/calvin_debug_dataset/validation --goals calvin_exp_configs_examples/guidance_goals/drawer_door_block.json \
        --output_folder datasets/guidance_conditions/my_goals

--recordings takes CALVIN folders and hdf5 files, also mixed, like the rollouts of run_dynaguide.py with --dataset_images.
The output folder holds a .hdf5 and a .png for each set, and a manifest with where each condition came from.
"""


def main(args):
    if os.path.isdir(args.output_folder) and len(os.listdir(args.output_folder)) > 0:
        sys.exit(f"Output folder {args.output_folder} is not empty, choose a new folder so nothing gets overwritten")

    recordings = read_recordings(args.recordings)
    print(f"Read {len(recordings)} recordings with {sum(len(recording) for recording in recordings)} frames")

    if args.goals is not None:
        with open(args.goals, "r") as f:
            counts = build_goal_conditions(recordings, json.load(f)["goals"], args.output_folder)
    else:
        counts = build_behavior_conditions(recordings, args.output_folder, args.behaviors, args.max_per_behavior)

    for name, count in counts.items():
        print(f"{name}: {count}" + ("  <- nothing matches, no file written" if count == 0 else ""))
    print(f"Saved to {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type = str, nargs = "+", required = True, help = "CALVIN folders with episode_*.npz, or hdf5 files")
    parser.add_argument("--output_folder", type = str, required = True, help = "new folder for the results")
    parser.add_argument("--goals", type = str, default = None, help = "goals file to find frames with. Without it, the recordings get cut into behaviors")
    parser.add_argument("--behaviors", type = str, nargs = "+", default = None, help = "behaviors to keep. By default all but other")
    parser.add_argument("--max_per_behavior", type = int, default = None, help = "limit of conditions per behavior, like the 20 of the paper")
    main(parser.parse_args())
