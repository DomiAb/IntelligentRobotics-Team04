import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))) # dynaguide_scripts, for recombine_conditions

from core.guidance_conditions import read_recordings
from recombine_conditions import DEFAULT_ENV_CONFIG, Simulator, make_env

"""
Renders recordings again in the simulator, frame by frame: the robot and the scene of each frame are put into the simulator as recorded
(no physics step) and the cameras render them. The images then look like those of the rollouts, which differ slightly from the
original CALVIN images (EGL renderer). Used to calibrate the switch of multi-goal guidance on real behavior sequences in the image
domain of the rollouts, see dynaguide_scripts/calibrate_goal_switching.py.

    python dynaguide_scripts/render_recording_in_sim.py --recordings datasets/calvin_debug_dataset/validation \
        --output datasets/calvin_debug_dataset_sim/validation.hdf5

The output is an hdf5 file in the format of the rollouts (one demo per recording, obs/states, obs/proprio, obs/third_person,
obs/eye_in_hand, actions), which read_recordings and create_guidance_conditions.py read like any rollout file.
"""


def main(args):
    if os.path.exists(args.output):
        sys.exit(f"{args.output} exists already, choose a new name so it isn't overwritten")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok = True)
    recordings = read_recordings(args.recordings)
    sim = Simulator(make_env(args.env_config))
    errors = list()
    with h5py.File(args.output, "w") as file:
        data = file.create_group("data")
        total = 0
        for number, recording in enumerate(recordings):
            frames = np.arange(0, len(recording), args.stride)
            images = {"third_person" : list(), "eye_in_hand" : list()}
            for frame in tqdm(frames, desc = recording.name):
                sim.set(recording.states[frame], recording.proprio[frame])
                errors.append(sim.robot_error(recording.proprio[frame]))
                for camera, image in sim.render().items():
                    images[camera].append(image)
            demo = data.create_group(f"demo_{number}")
            demo.create_dataset("actions", data = recording.actions[frames])
            demo.create_dataset("obs/states", data = recording.states[frames])
            demo.create_dataset("obs/proprio", data = recording.proprio[frames])
            for camera, stack in images.items():
                demo.create_dataset(f"obs/{camera}", data = np.stack(stack).astype(np.uint8), compression = "gzip")
            demo.attrs["source"] = recording.name
            demo.attrs["stride"] = args.stride
            demo.attrs["num_samples"] = len(frames)
            total += len(frames)
        data.attrs["total"] = total
    print(f"Wrote {total} frames to {args.output}. Robot placed within {np.max(errors):.4f} (max over tcp position and arm joints)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", nargs = "+", required = True, help = "CALVIN folders or hdf5 files")
    parser.add_argument("--output", required = True, help = "new hdf5 file")
    parser.add_argument("--stride", type = int, default = 1, help = "render every n-th frame")
    parser.add_argument("--env_config", default = DEFAULT_ENV_CONFIG)
    main(parser.parse_args())
