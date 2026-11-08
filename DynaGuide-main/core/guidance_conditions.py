import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Callable

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from core.calvin_behaviors import ARTICULATED

"""
Creates guidance conditions (hdf5 files for pos_examples and neg_examples) from recorded data. A source becomes a list of Recording,
and everything after that doesn't know where the data came from:

    read_recordings        CALVIN folder with .npz files, or hdf5 file (rollouts of run_dynaguide.py with --dataset_images, condition files)
          |
    segment_recording      cuts a recording into behaviors, like the authors did for the guidance conditions of the paper
    find_frames            finds the frames that match a description of a state, for goals that aren't a whole behavior
          |
    write_conditions       writes the existing format of the guidance conditions

DynaGuide only uses the last frame of each demo of a condition file: third_person image, proprio and the action, see calculate_classifier_guidance.
"""

CAMERAS = ["third_person", "eye_in_hand"]
CALVIN_CAMERA_KEYS = {"third_person" : "rgb_static", "eye_in_hand" : "rgb_gripper"}
STATE_DIM, PROPRIO_DIM, ACTION_DIM = 24, 15, 7

# --- reading the sources ------------------------------------------------------------------------------------------------------------

@dataclass
class Recording:
    """
    One continuous recording. Frame t has the observations at t and the action taken from there.
    - load_images: function (camera, frames) -> images [T, H, W, 3] as uint8, so they only get loaded when they are needed
    """
    name: str
    states: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    cameras: list
    load_images: Callable

    def __post_init__(self):
        if "third_person" not in self.cameras:
            raise ValueError(f"{self.name} has no third_person images, which is what the guidance uses. For rollouts, record them with --dataset_images")
        for values, dim, label in [(self.states, STATE_DIM, "states"), (self.proprio, PROPRIO_DIM, "proprio"), (self.actions, ACTION_DIM, "actions")]:
            if values.ndim != 2 or values.shape[1] != dim or len(values) != len(self.states):
                raise ValueError(f"{self.name}: {label} has shape {values.shape}, expected [{len(self.states)}, {dim}]")

    def __len__(self):
        return len(self.states)


def read_calvin_folder(folder):
    """Reads a split of a CALVIN dataset: episode_XXXXXXX.npz per frame, and ep_start_end_ids.npy with the continuous recordings."""
    recordings = list()
    split = os.path.basename(os.path.normpath(folder))
    for number, (first, last) in enumerate(np.load(os.path.join(folder, "ep_start_end_ids.npy"))):
        files = [os.path.join(folder, f"episode_{frame:07d}.npz") for frame in range(first, last + 1)]
        missing = [file for file in files if not os.path.exists(file)]
        if len(missing) > 0:
            raise FileNotFoundError(f"{folder} has {len(missing)} missing frames in a recording, the first is {missing[0]}")

        frames = [np.load(file) for file in files]
        states = np.array([frame["scene_obs"] for frame in frames], dtype = np.float64)
        proprio = np.array([frame["robot_obs"] for frame in frames], dtype = np.float64)
        actions = np.clip(np.array([frame["rel_actions"] for frame in frames], dtype = np.float64), -1, 1) # the authors clip them too

        def load_images(camera, wanted, files = files):
            return np.stack([np.load(files[frame])[CALVIN_CAMERA_KEYS[camera]] for frame in wanted])

        recordings.append(Recording(f"{split}_ep{number}", states, proprio, actions, list(CAMERAS), load_images))
    return recordings


def read_hdf5_file(path):
    """Reads an hdf5 file in the robomimic format: the rollouts of run_dynaguide.py or guidance conditions. Each demo is a recording."""
    recordings = list()
    source = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as file:
        demos = sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1]))
        for demo in demos:
            group = file["data"][demo]
            if "actions" not in group or "obs" not in group: # an empty last demo, see calvin_to_labeled_hdf5.py
                continue
            missing = [key for key in ["states", "proprio"] if key not in group["obs"]]
            if len(missing) > 0:
                raise KeyError(f"{path} {demo} has no {missing}. Run the rollouts with --dataset_obs")

            def load_images(camera, wanted, demo = demo):
                with h5py.File(path, "r") as reader:
                    images = reader["data"][demo]["obs"][camera]
                    return np.stack([images[frame] for frame in wanted])

            cameras = [camera for camera in CAMERAS if camera in group["obs"]]
            recordings.append(Recording(f"{source}_{demo}", group["obs/states"][()], group["obs/proprio"][()], group["actions"][()], cameras, load_images))
    return recordings


def read_recordings(paths):
    """Reads any mix of CALVIN folders and hdf5 files."""
    recordings = list()
    for path in paths:
        if os.path.isdir(path):
            if len(glob.glob(os.path.join(path, "episode_*.npz"))) == 0:
                raise FileNotFoundError(f"{path} has no episode_*.npz files. Pass a split of the dataset, like its validation folder")
            recordings += read_calvin_folder(path)
        elif path.endswith((".hdf5", ".h5")):
            recordings += read_hdf5_file(path)
        else:
            raise ValueError(f"{path} is neither a CALVIN folder nor an hdf5 file")
    if len(recordings) == 0:
        raise ValueError(f"Found no recordings in {paths}")
    return recordings

# --- cutting a recording into behaviors ---------------------------------------------------------------------------------------------

@dataclass
class Segment:
    """Frames first to last of a recording, last included. The last frame is what the guidance uses."""
    recording: Recording
    first: int
    last: int
    behavior: str

    @property
    def label(self):
        return f"{self.recording.name}:{self.last}"


# how much a change of each part counts for detecting that something moves
MOTION_SCALE = {"sliding_door" : 10, "drawer" : 10, "button" : 10, "switch" : 10, "lightbulb" : 0, "green_light" : 0,
                "red_block" : 1, "blue_block" : 1, "pink_block" : 1}
MOTION_EPSILON = 0.001
MIN_MOTION_STEPS = 10
MIN_SEGMENT_LENGTH = 16
BLOCK_TOUCH_DISTANCE = 0.05
JOINT_INDEX = {"sliding_door" : 0, "drawer" : 1, "switch" : 3, "green_light" : 5}


def _parts(state):
    return ({"sliding_door" : state[0], "drawer" : state[1], "button" : state[2], "switch" : state[3], "lightbulb" : state[4], "green_light" : state[5],
             "red_block" : state[6:9], "blue_block" : state[12:15], "pink_block" : state[18:21]},
            {"red" : R.from_euler("XYZ", state[9:12]).as_matrix(), "blue" : R.from_euler("XYZ", state[15:18]).as_matrix(), "pink" : R.from_euler("XYZ", state[21:]).as_matrix()})


def _label_behavior(start, end, moving, delta):
    """
    Names the behavior from the start and end state of a segment, in the order of calvin_to_labeled_hdf5.py.
    The displace check is copied as it is: the comparison is inside the norm, so it only finds blocks that moved towards larger x, y or z.
    A lift is what calvin_to_labeled_hdf5.py calls the moment a block near the robot starts moving. The block isn't up yet in that frame.
    """
    for index, low, high, rising, falling in [(5, 0.2, 0.8, "button_on", "button_off"), (3, 0.01, 0.07, "switch_on", "switch_off"),
                                              (1, 0.05, 0.10, "drawer_open", "drawer_close"), (0, 0.05, 0.25, "door_left", "door_right")]:
        name = {0 : "sliding_door", 1 : "drawer", 3 : "switch", 5 : "green_light"}[index]
        if end[name] > high and start[name] < low:
            return rising
        if end[name] < low and start[name] > high:
            return falling
    for color in ["red", "blue", "pink"]:
        if np.linalg.norm(end[color + "_block"] - start[color + "_block"] > 0.01):
            return color + "_displace"
    if moving:
        for color in ["red", "pink", "blue"]:
            if delta[color + "_block"] > MOTION_EPSILON:
                return color + "_lift"
    return "other"


def segment_recording(recording, keep_unfinished = True):
    """
    Cuts a recording into behaviors like calvin_to_labeled_hdf5.py: a segment ends when the things that moved come to rest, or when a block
    near the robot starts moving. All frames since the last cut belong to the segment.
    - keep_unfinished: also keep what is going on at the end of the recording if it has a name. A rollout stops the moment its goal is
      reached, so here its goal would be lost. calvin_to_labeled_hdf5.py doesn't do this, because its recordings don't end at anything special.
    """
    segments = list()
    waiting, moving = True, False
    move_steps, first = 0, 0
    last_state, last_rotation = _parts(recording.states[0])
    start_state = last_state
    for frame in range(len(recording)):
        state, rotation = _parts(recording.states[frame])
        delta = {name : MOTION_SCALE[name] * float(np.linalg.norm(state[name] - last_state[name])) for name in state}
        delta_rotation = [np.linalg.norm(R.from_matrix(rotation[color] @ np.linalg.inv(last_rotation[color])).as_euler("XYZ")) for color in rotation]
        any_motion = max(delta.values()) > MOTION_EPSILON or max(delta_rotation) > MOTION_EPSILON

        cut = False
        if moving:
            move_steps += 1
        if waiting and any_motion:
            moving, waiting = True, False
            if max(delta[color + "_block"] for color in ["red", "blue", "pink"]) > MOTION_EPSILON:
                tcp = recording.proprio[frame, :3]
                cut = min(np.linalg.norm(tcp - state[color + "_block"]) for color in ["red", "blue", "pink"]) < BLOCK_TOUCH_DISTANCE
        elif moving and not any_motion:
            waiting, moving = True, False
            cut = move_steps >= MIN_MOTION_STEPS and frame - first >= MIN_SEGMENT_LENGTH
            move_steps = 0
            if state["green_light"] != start_state["green_light"]: # always cut when the light changes
                cut = True

        if cut:
            segments.append(Segment(recording, first, frame, _label_behavior(start_state, state, moving, delta)))
            start_state = state
            first = frame + 1
        last_state, last_rotation = state, rotation

    if keep_unfinished and first < len(recording):
        behavior = _label_behavior(start_state, state, moving, delta)
        if behavior != "other":
            segments.append(Segment(recording, first, len(recording) - 1, behavior))
    return segments

# --- finding frames by a description of the state -------------------------------------------------------------------------------------

JOINT_WORDS = {"sliding_door" : ("right", "left"), "drawer" : ("closed", "open"), "switch" : ("off", "on"), "green_light" : ("off", "on")}
# what the words mean: the thresholds of the behaviors, in the same order as JOINT_INDEX
JOINT_WORD_RANGES = {name : {JOINT_WORDS[name][0] : (-math.inf, low), JOINT_WORDS[name][1] : (high, math.inf)}
                     for name, (_, low, high, _, _) in zip(["sliding_door", "drawer", "switch", "green_light"], ARTICULATED)}

BLOCK_POSITION = {"red" : 6, "blue" : 12, "pink" : 18}
BLOCK_REST_Z = (0.362, 0.46) # resting in the drawer, or on the table and in the cabinet
HELD_DISTANCE, HELD_WIDTH, OPEN_WIDTH = 0.04, 0.06, 0.07 # the gripper is 0.08 wide when open and around 0.04 on a block
CABINET_MIDDLE_X, CABINET_FRONT_Y = -0.06, 0.03
GRIPPER_WIDTH, GRIPPER_ACTION = 6, 14 # index in the proprio
DEFAULT_TCP_RADIUS = 0.08

# Where the gripper is when it works on something, averaged from the end frames of the behaviors in the CALVIN debug set. Handles move with their joint
def _anchor_position(name, states):
    count = len(states)
    if name == "drawer_handle":
        return np.stack([np.full(count, 0.175), -0.205 - states[:, 1], np.full(count, 0.40)], axis = 1)
    if name == "door_handle":
        return np.stack([0.04 - states[:, 0], np.full(count, -0.01), np.full(count, 0.54)], axis = 1)
    if name == "switch":
        return np.tile([0.285, 0.03, 0.57], (count, 1))
    if name == "button":
        return np.tile([-0.105, -0.115, 0.52], (count, 1))
    if name in BLOCK_POSITION:
        return states[:, BLOCK_POSITION[name] : BLOCK_POSITION[name] + 3]
    if isinstance(name, list) and len(name) == 3:
        return np.tile(name, (count, 1))
    raise ValueError(f"Unknown tcp_near {name}, choose from {TCP_ANCHORS + list(BLOCK_POSITION)} or give [x, y, z]")

TCP_ANCHORS = ["drawer_handle", "door_handle", "switch", "button"]


def _block_mask(color, word, recording):
    position = recording.states[:, BLOCK_POSITION[color] : BLOCK_POSITION[color] + 3]
    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    resting_height = np.abs(z - BLOCK_REST_Z[1]) < 0.015
    in_cabinet = resting_height & (y > CABINET_FRONT_Y)
    near_tcp = np.linalg.norm(recording.proprio[:, :3] - position, axis = 1) < HELD_DISTANCE
    held = near_tcp & (recording.proprio[:, GRIPPER_WIDTH] < HELD_WIDTH) & (recording.proprio[:, GRIPPER_ACTION] < 0)
    masks = {"drawer" : z < 0.40,
             "table" : resting_height & (y <= CABINET_FRONT_Y),
             "cabinet" : in_cabinet,
             "cabinet_left" : in_cabinet & (x < CABINET_MIDDLE_X),
             "cabinet_right" : in_cabinet & (x >= CABINET_MIDDLE_X),
             "held" : held,
             "lifted" : held & (np.abs(z - BLOCK_REST_Z[0]) > 0.02) & (np.abs(z - BLOCK_REST_Z[1]) > 0.02)}
    if word not in masks:
        raise ValueError(f"Unknown state '{word}' for block {color}, choose from {list(masks)}")
    return masks[word]


def find_frames(recording, where):
    """
    Returns a mask of the frames that match all of this description. See calvin_exp_configs_examples/guidance_goals for an example.
    - sliding_door, drawer, switch, green_light: the words of env_setup, or a range [low, high]
    - blocks: {"red" : "drawer"} with one of drawer, table, cabinet, cabinet_left, cabinet_right, held, lifted
    - tcp_near: drawer_handle, door_handle, switch, button, a block color or [x, y, z]. tcp_radius is how close, 0.08 by default
    - gripper: open or closed
    """
    unknown = [key for key in where if key not in list(JOINT_WORD_RANGES) + ["blocks", "tcp_near", "tcp_radius", "gripper"]]
    if len(unknown) > 0:
        raise ValueError(f"Unknown keys {unknown} in where, choose from {list(JOINT_WORD_RANGES) + ['blocks', 'tcp_near', 'tcp_radius', 'gripper']}")

    mask = np.ones(len(recording), dtype = bool)
    for name, word_ranges in JOINT_WORD_RANGES.items():
        if name not in where:
            continue
        wanted = where[name]
        if isinstance(wanted, str):
            if wanted not in word_ranges:
                raise ValueError(f"where {name}: unknown word '{wanted}', choose from {list(word_ranges)} or give [low, high]")
            wanted = word_ranges[wanted]
        low, high = wanted
        mask &= (recording.states[:, JOINT_INDEX[name]] >= low) & (recording.states[:, JOINT_INDEX[name]] <= high)

    for color, word in where.get("blocks", {}).items():
        if color not in BLOCK_POSITION:
            raise ValueError(f"where blocks: unknown block '{color}', choose from {list(BLOCK_POSITION)}")
        mask &= _block_mask(color, word, recording)

    if "tcp_near" in where:
        anchor = _anchor_position(where["tcp_near"], recording.states)
        mask &= np.linalg.norm(recording.proprio[:, :3] - anchor, axis = 1) < where.get("tcp_radius", DEFAULT_TCP_RADIUS)

    if "gripper" in where:
        if where["gripper"] not in ["open", "closed"]:
            raise ValueError(f"where gripper: unknown word '{where['gripper']}', choose from open, closed")
        width = recording.proprio[:, GRIPPER_WIDTH]
        mask &= width > OPEN_WIDTH if where["gripper"] == "open" else width < HELD_WIDTH
    return mask


def select_frames(recordings, masks, n = None, min_gap = 15, exclude = ()):
    """
    Picks frames from the matches so that they differ from each other, because neighbouring frames look the same and add nothing to the guidance.
    First the last frame of every separate match, then frames further back in the matches, each min_gap apart. Without n, only the first.
    - exclude: labels of frames to leave out, like "validation_ep0:1234" as written on the preview
    Returns the picks as [(recording, frame)] in the order of the recordings
    """
    tiers = list() # tiers[k] holds the frames that are k * min_gap before the end of their match
    for recording, mask in zip(recordings, masks):
        frames = np.flatnonzero(mask)
        if len(frames) == 0:
            continue
        for match in np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1):
            for tier, position in enumerate(range(len(match) - 1, -1, -min_gap)):
                if f"{recording.name}:{match[position]}" not in exclude:
                    while len(tiers) <= tier:
                        tiers.append(list())
                    tiers[tier].append((recording, int(match[position])))

    picks = list()
    for tier in tiers:
        remaining = None if n is None else n - len(picks)
        if remaining is not None and remaining <= 0:
            break
        if remaining is None or remaining >= len(tier):
            picks += tier
        else: # spread what is left over the recordings
            picks += [tier[index] for index in np.linspace(0, len(tier) - 1, remaining).round().astype(int)]
        if n is None:
            break
    order = {recording.name : index for index, recording in enumerate(recordings)}
    return sorted(picks, key = lambda pick: (order[pick[0].name], pick[1]))

# --- writing the conditions -----------------------------------------------------------------------------------------------------------

def write_conditions(path, segments):
    """Writes segments as demos of a guidance condition file, in the format of calvin_to_labeled_hdf5.py. Stops instead of overwriting."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists, choose a new folder so it isn't overwritten")
    os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        total = 0
        for number, segment in enumerate(segments):
            recording, frames = segment.recording, np.arange(segment.first, segment.last + 1)
            demo = data.create_group(f"demo_{number}")
            demo.create_dataset("actions", data = recording.actions[frames])
            demo.create_dataset("obs/states", data = recording.states[frames])
            demo.create_dataset("obs/proprio", data = recording.proprio[frames])
            for camera in recording.cameras:
                demo.create_dataset(f"obs/{camera}", data = recording.load_images(camera, frames), compression = "gzip")
            dones = np.zeros(len(frames))
            dones[-1] = 1
            demo.create_dataset("rewards", data = np.zeros(len(frames)))
            demo.create_dataset("dones", data = dones)
            demo.attrs["behavior"] = segment.behavior
            demo.attrs["num_samples"] = len(frames)
            demo.attrs["source"] = segment.label
            total += len(frames)
        data.attrs["total"] = total


def save_preview(path, segments, size = 200):
    """Saves the last frame of each condition, which is what the guidance uses. The label on it is what goals can exclude."""
    tiles = list()
    for segment in segments:
        tile = cv2.resize(segment.recording.load_images("third_person", [segment.last])[0], (size, size))
        for color, thickness in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
            cv2.putText(tile, segment.label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, thickness, cv2.LINE_AA)
        tiles.append(tile)
    columns = int(np.ceil(np.sqrt(len(tiles))))
    tiles += [np.zeros_like(tiles[0])] * (columns * int(np.ceil(len(tiles) / columns)) - len(tiles))
    rows = [np.concatenate(tiles[i:i + columns], axis = 1) for i in range(0, len(tiles), columns)]
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, axis = 0), cv2.COLOR_RGB2BGR))


def _describe(segment):
    state, proprio = segment.recording.states[segment.last], segment.recording.proprio[segment.last]
    return {"source" : segment.label, "first_frame" : segment.first, "last_frame" : segment.last, "behavior" : segment.behavior,
            "sliding_door" : round(float(state[0]), 3), "drawer" : round(float(state[1]), 3), "switch" : round(float(state[3]), 3),
            "green_light" : round(float(state[5]), 3), "tcp" : [round(float(x), 3) for x in proprio[:3]],
            **{color : [round(float(x), 3) for x in state[index:index + 3]] for color, index in BLOCK_POSITION.items()}}

# --- the pipeline --------------------------------------------------------------------------------------------------------------------

def _save_condition_set(output_folder, name, segments, manifest):
    write_conditions(os.path.join(output_folder, f"{name}.hdf5"), segments)
    save_preview(os.path.join(output_folder, f"{name}.png"), segments)
    manifest[name] = [_describe(segment) for segment in segments]


def build_behavior_conditions(recordings, output_folder, behaviors = None, max_per_behavior = None):
    """
    Writes one condition file per behavior, with the segments of the recordings. Spreads over the recordings if max_per_behavior limits them.
    - behaviors: names to keep, None means all but "other"
    Returns the number of conditions per behavior.
    """
    found = dict()
    for recording in recordings:
        for segment in segment_recording(recording):
            if (behaviors is None and segment.behavior != "other") or (behaviors is not None and segment.behavior in behaviors):
                found.setdefault(segment.behavior, list()).append(segment)

    manifest = dict()
    for behavior, segments in sorted(found.items()):
        if max_per_behavior is not None and len(segments) > max_per_behavior:
            segments = [segments[index] for index in np.linspace(0, len(segments) - 1, max_per_behavior).round().astype(int)]
        _save_condition_set(output_folder, behavior, segments, manifest)
    _save_manifest(output_folder, "behaviors", manifest)
    return {behavior : len(segments) for behavior, segments in manifest.items()}


def build_goal_conditions(recordings, goals, output_folder, default_history = 16):
    """
    Writes one condition file per goal of a goals file. A condition is a chosen frame and the frames before it.
    - goals: list of {"name", "where", optional "n", "min_gap", "exclude", "history"}
    Returns the number of conditions per goal, which is 0 if nothing matches.
    """
    unknown = [key for goal in goals for key in goal if key not in ["name", "where", "n", "min_gap", "exclude", "history"]]
    if len(unknown) > 0:
        raise ValueError(f"Unknown keys {unknown} in the goals, choose from name, where, n, min_gap, exclude, history")
    names = [goal["name"] for goal in goals]
    if len(set(names)) != len(names):
        raise ValueError(f"Two goals have the same name: {names}")

    # search everything first, so a typo in the last goal stops before anything gets written
    selected = list()
    for goal in goals:
        masks = [find_frames(recording, goal["where"]) for recording in recordings]
        picks = select_frames(recordings, masks, goal.get("n"), goal.get("min_gap", 15), goal.get("exclude", ()))
        history = goal.get("history", default_history)
        selected.append([Segment(recording, max(0, frame - history + 1), frame, goal["name"]) for recording, frame in picks])

    manifest = dict()
    counts = dict()
    for goal, segments in zip(goals, selected):
        counts[goal["name"]] = len(segments)
        if len(segments) > 0:
            _save_condition_set(output_folder, goal["name"], segments, manifest)
    _save_manifest(output_folder, "goals", manifest)
    return counts


def _save_manifest(output_folder, kind, manifest):
    os.makedirs(output_folder, exist_ok = True)
    with open(os.path.join(output_folder, f"manifest_{kind}.json"), "w") as f:
        json.dump(manifest, f, indent = 2)
