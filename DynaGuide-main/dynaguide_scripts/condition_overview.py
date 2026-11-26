import argparse
import glob
import json
import os

import cv2
import h5py
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_utils import condition_files

"""
A picture of every guidance condition a run used, so it can be looked at later next to the run's videos: conditions.png in the run
folder. run_dynaguide.py writes it at the start of every run. For runs made before, give their folders (or folders above them):

    python dynaguide_scripts/condition_overview.py --results results/random_scenes_singles results/random_scenes_chain

One block per file: the goal conditions of each goal step (pos_examples), and the images of its switch (before_examples, goal_examples,
contrast_examples), or pos_examples and neg_examples of a single-goal run. Each image is the last frame of a condition, as the guidance
and the switch use it, numbered so a condition and its before-twin can be matched. A file that isn't there any more is listed as missing.
"""

SIZE = 128
PER_ROW = 12
FILENAME = "conditions.png"
SWITCH_KEYS = ["before_examples", "goal_examples", "contrast_examples"]


def last_frames(path, camera = "third_person"):
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key = lambda name: int(name.split("_")[-1]))
        return [f["data"][demo]["obs"][camera][-1] for demo in demos]


def config_blocks(config):
    """(label, condition spec) of every set of images the config uses."""
    blocks = list()
    if "goal_sequence" in config:
        for number, step in enumerate(config["goal_sequence"]):
            name = step.get("name", f"step_{number}")
            blocks.append((f"step {number + 1} {name}: goal conditions, scale {step.get('scale')}, alpha {step.get('alpha')}", step.get("pos_examples")))
            for key in SWITCH_KEYS:
                if step.get("switch", dict()).get(key) is not None:
                    blocks.append((f"step {number + 1} {name}: switch {key}", step["switch"][key]))
    else:
        blocks.append(("goal conditions (pos_examples)", config.get("pos_examples")))
        if config.get("use_neg") and config.get("neg_examples") is not None:
            blocks.append(("neg_examples", config["neg_examples"]))
    portable = lambda spec: spec.replace("\\", "/") if isinstance(spec, str) else [entry.replace("\\", "/") for entry in spec] # a run from Windows
    return [(label, portable(spec)) for label, spec in blocks if spec is not None]


def header(text, width):
    line = np.full((20, width, 3), 255, np.uint8)
    cv2.putText(line, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return line


def save_condition_overview(config, path):
    """Writes the picture of every condition of config to path. Returns the path, or None if no condition file was found."""
    width, lines = SIZE * PER_ROW, list()
    for label, spec in config_blocks(config):
        try:
            files = condition_files(spec)
        except (FileNotFoundError, ValueError):
            lines.append(header(f"{label}: missing {spec}", width))
            continue
        images = [image for file in files for image in last_frames(file)]
        lines.append(header(f"{label} ({len(images)}) | {', '.join(os.path.relpath(file) for file in files)}"[:170], width))
        tiles = list()
        for number, image in enumerate(images):
            tile = cv2.resize(np.ascontiguousarray(image), (SIZE, SIZE))
            cv2.putText(tile, str(number), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        for start in range(0, len(tiles), PER_ROW):
            row = tiles[start : start + PER_ROW]
            row += [np.zeros((SIZE, SIZE, 3), np.uint8)] * (PER_ROW - len(row))
            lines.append(np.concatenate(row, axis = 1))
    if len(lines) == 0:
        return None
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(lines, axis = 0), cv2.COLOR_RGB2BGR))
    return path


def main(args):
    folders = sorted({os.path.dirname(path) for root in args.results
                      for path in glob.glob(os.path.join(root, "**", "config.json"), recursive = True)})
    for folder in folders:
        with open(os.path.join(folder, "config.json")) as f:
            config = json.load(f)
        if "goal_sequence" not in config and "pos_examples" not in config:
            continue
        written = save_condition_overview(config, os.path.join(folder, FILENAME))
        print(f"{folder}: {'wrote ' + FILENAME if written else 'no condition files found'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs = "+", required = True, help = "run folders (with config.json), or folders above them")
    main(parser.parse_args())
