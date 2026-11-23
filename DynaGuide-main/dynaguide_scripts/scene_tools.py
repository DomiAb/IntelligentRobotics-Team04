import itertools

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_utils import BLOCK_INDEX, HIDDEN_POS, PLACES, STATE_WORDS, parse_block
from core.guidance_conditions import CABINET_MIDDLE_X, TABLE_THIRDS, Recording, Segment, _anchor_position, write_conditions
from recombine_conditions import ANCHOR, DRAWER_BLOCK, STATE_INDEX, joint_word

"""
What extract_block_poses.py (the block base sets) and make_task.py (the conditions of one task) share: the scene of a condition as joint
words, putting a block of one color where another one was, and checking that a rendered condition doesn't put anything inside anything.
"""

JOINTS = ["drawer", "sliding_door", "switch", "green_light"] # in the order of the file names
SHORT = {"sliding_door" : "door"}
GOAL_WORD = {"drawer" : "open", "sliding_door" : "left", "switch" : "on", "green_light" : "on"} # the high end of each joint
PENETRATION_TOLERANCE = 0.002


def scene_name(state):
    """drawer_open__door_left__switch_off__green_light_off, the file name of a scene in the base sets."""
    return "__".join(f"{SHORT.get(joint, joint)}_{joint_word(joint, state[STATE_INDEX[joint]])}" for joint in JOINTS)


def joint_combinations(fixed = ()):
    """Every {joint : word} of the joints that aren't fixed."""
    free = [joint for joint in JOINTS if joint not in fixed]
    return [dict(zip(free, words)) for words in itertools.product(*[sorted(STATE_WORDS[joint]) for joint in free])]


def block_area(color, state):
    """Where a resting block is: table/left, table/center, table/right, drawer, cabinet/left, cabinet/right, or None."""
    x, y, z = state[BLOCK_INDEX[color] : BLOCK_INDEX[color] + 3]
    if x > 2.0:
        return None
    if DRAWER_BLOCK["x"][0] < x < DRAWER_BLOCK["x"][1] and z < DRAWER_BLOCK["z_below"]:
        return "drawer"
    if y > 0.03: # the shelf behind the sliding door, see CABINET_FRONT_Y
        return "cabinet/left" if x < CABINET_MIDDLE_X else "cabinet/right"
    return "table/left" if x < TABLE_THIRDS[0] else "table/center" if x < TABLE_THIRDS[1] else "table/right"


# where a block lies after the task let go of it in an area, unless the task says otherwise. Spots of calvin_utils.PLACES
AREA_SPOT = {"table/left" : "table_left", "table/center" : "table_center", "table/right" : "table_right", "drawer" : "drawer_center",
             "cabinet/left" : "cabinet_left_center", "cabinet/right" : "cabinet_right_center"}


def put_block_at_place(state, color, place):
    """The block at a place of calvin_utils.PLACES (a spot), following the drawer if it is a drawer place."""
    low, _, inside, yaw = parse_block(color, place)
    index = BLOCK_INDEX[color]
    state[index : index + 6] = np.concatenate([low, [0.0, 0.0, yaw if yaw is not None else 0.0]])
    if inside:
        state[index + 1] -= state[STATE_INDEX["drawer"]]


def hide_block(state, color):
    index = BLOCK_INDEX[color]
    state[index : index + 3], state[index + 3 : index + 6] = HIDDEN_POS[color], 0.0


def long_side_along_gripper(block_euler, tcp_euler):
    """
    The block's rotation with its local x axis (the long side) set to whichever of its axes is closest to the gripper's x axis. All
    blocks are 4 cm wide and high, only the length differs, and the gripper closes on two 4 cm faces with the long side along it. So a
    pose of one block fits any other, turned like this. For the cube-shaped blue block it changes nothing visible.
    """
    block = Rotation.from_euler("xyz", block_euler).as_matrix()
    gripper_x = Rotation.from_euler("xyz", tcp_euler).as_matrix()[:, 0]
    axis = int(np.argmax(np.abs(block.T @ gripper_x)))
    order = {0 : [0, 1, 2], 1 : [1, 2, 0], 2 : [2, 0, 1]}[axis] # cyclic, so the frame stays right-handed
    return Rotation.from_matrix(block[:, order]).as_euler("xyz")


def move_block(state, source_state, source_color, color, tcp_euler):
    """Puts block color where source_color is in source_state, turned to fit the gripper if the color changes."""
    source, target = BLOCK_INDEX[source_color], BLOCK_INDEX[color]
    state[target : target + 3] = source_state[source : source + 3]
    euler = source_state[source + 3 : source + 6]
    state[target + 3 : target + 6] = euler if source_color == color else long_side_along_gripper(euler, tcp_euler)


def held_joint_moved(pose_state, tcp, state, radius = 0.1):
    """A joint that the gripper is at in the pose and that state moves to the other side, or None."""
    for joint, anchor in ANCHOR.items():
        near = np.linalg.norm(_anchor_position(anchor, pose_state[None])[0] - tcp) < radius
        if near and joint_word(joint, state[STATE_INDEX[joint]]) != joint_word(joint, pose_state[STATE_INDEX[joint]]):
            return joint
    return None


def new_contacts(sim, pose_state, state, proprio, renamed = None):
    """
    What is deeper inside something in state than in the pose (pose_state with the same robot). renamed: {color in state : color in the
    pose} for a block that took another block's place. Returns a description of the first problem, or None.
    """
    renamed = renamed or dict()
    sim.set(pose_state, proprio)
    before = sim.contacts()
    sim.set(state, proprio)
    for key, depth in sim.contacts().items():
        original = (renamed.get(key[0], key[0]), *key[1:])
        if depth > before.get(original, 0.0) + PENETRATION_TOLERANCE:
            return sim.describe(key)
    return None


def read_condition_poses(path):
    """The last frame of every demo of a condition file, with its attributes."""
    poses = list()
    with h5py.File(path, "r") as file:
        for demo in sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1])):
            group = file["data"][demo]
            poses.append({"states" : group["obs/states"][-1].astype(np.float64), "proprio" : group["obs/proprio"][-1].astype(np.float64),
                          "action" : group["actions"][-1].astype(np.float64), "attrs" : dict(group.attrs), "file" : f"{path}:{demo}"})
    return poses


def write_rendered(path, rendered, behavior, attributes = None):
    """rendered: list of (images, state, pose). Writes them as single-frame conditions, with extra demo attributes per condition."""
    segments = list()
    for images, state, pose in rendered:
        source = str(pose["attrs"].get("source", pose["file"]))
        recording = Recording(source, state[None], pose["proprio"][None], pose["action"][None], ["third_person", "eye_in_hand"],
                              lambda camera, frames, images = images: np.stack([images[camera] for _ in frames]))
        segments.append(Segment(recording, 0, 0, behavior))
    write_conditions(path, segments)
    if attributes is not None:
        with h5py.File(path, "a") as file:
            for number, extra in enumerate(attributes):
                for key, value in extra.items():
                    file["data"][f"demo_{number}"].attrs[key] = value
    return segments


def save_rows(path, rows, size = 128, per_row = 8):
    """One row of images per (label, images), the label above it."""
    lines = list()
    for label, images in rows:
        tiles = [cv2.resize(image, (size, size)) for image in images[:per_row]]
        tiles += [np.zeros((size, size, 3), np.uint8)] * (per_row - len(tiles))
        header = np.full((18, size * per_row, 3), 255, np.uint8)
        cv2.putText(header, f"{label} ({len(images)})", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        lines += [header, np.concatenate(tiles, axis = 1)]
    if len(lines) > 0:
        cv2.imwrite(path, cv2.cvtColor(np.concatenate(lines, axis = 0), cv2.COLOR_RGB2BGR))
