import argparse
import copy
import itertools
import json
import os
import sys

import cv2
import h5py
import numpy as np

from calvin_env.envs.play_table_env import PlayTableSimEnv
from core.calvin_behaviors import ARTICULATED
from core.calvin_utils import BLOCK_INDEX, HIDDEN_POS, STATE_WORDS
from core.guidance_conditions import Recording, Segment, _anchor_position, save_preview, write_conditions

"""
Makes more guidance conditions out of few: puts the robot of each existing condition back into the simulator exactly as it was
at the goal, and renders it again with every combination of the other joints. See section 6 of "Create guidance conditions.ipynb".

    python recombine_conditions.py --output_folder datasets/guidance_conditions/recombined_debug

By default this takes the drawer, door and switch conditions of datasets/guidance_conditions/debug_all. Pass others with
--conditions, any condition file whose behavior is one of drawer_open, drawer_close, door_left, door_right, switch_on, switch_off.

What changes and what doesn't, for each condition (its last frame, which is what the guidance uses):
- the robot, its action and the joint of the goal stay as they are, like the drawer in a drawer_open condition
- a joint the gripper is holding or very close to stays as well. A combination that would need to move it is skipped for this condition
- every other joint of --joints (drawer, sliding door, switch and green light by default) takes each of its two states. The yellow
  light follows the switch, the green light is set directly, as the button would leave it
- blocks stay where they were. Blocks in the drawer move with it. A block that would end up inside the table, a door, the drawer or the
  robot is removed from the scene, because it blocks the movement
- a combination where the robot would end up inside a part of the table is skipped for this condition
- with --hide_blocks, all blocks are parked outside the camera view instead

The output folder has one folder per behavior with one condition file per combination, each with a preview:
    drawer_close/drawer_closed__door_left__switch_off__green_light_on.hdf5   drawer_close with door left, switch off and green light on
    drawer_close/overview.png                                   every condition in every combination, black where skipped
    manifest.json                                               what was done to each condition, and what was skipped and why

All images come from the simulator, also for the combination the condition already had. So the files of one run are consistent with each
other and with what the policy sees in the simulator, but differ slightly from the original images of the CALVIN data, which used EGL.
"""

REPO_FOLDER = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV_CONFIG = os.path.join(REPO_FOLDER, "configs", "calvin.json")
DEFAULT_CONDITIONS = [os.path.join("datasets", "guidance_conditions", "debug_all", f"{goal}.hdf5")
                      for goal in ["drawer_open", "drawer_close", "door_left", "door_right", "switch_on", "switch_off"]]

STATE_INDEX = {"sliding_door" : 0, "drawer" : 1, "switch" : 3, "green_light" : 5}
LIGHTBULB_INDEX = 4
ANCHOR = {"sliding_door" : "door_handle", "drawer" : "drawer_handle", "switch" : "switch", "green_light" : "button"} # where the gripper holds the joint, the button toggles the green light
SHORT_NAMES = {"sliding_door" : "door"} # for the file names
GOALS = {(rising if end == "high" else falling) : (joint, end) for joint, (_, _, _, rising, falling) in zip(["sliding_door", "drawer", "switch"], ARTICULATED)
         for end in ["high", "low"]}
PENETRATION_TOLERANCE = 0.002 # deeper than the condition already had, in meters, counts as inside
DRAWER_BLOCK = dict(x = (0.0, 0.36), z_below = 0.40) # a block this low inside the drawer's x range sits in the drawer


def make_env(env_config_path):
    """The simulator without a window. The images come from the CPU renderer, like on the workstation."""
    with open(env_config_path, "r") as f:
        env_kwargs = copy.deepcopy(json.load(f)["env_kwargs"])
    env_kwargs.update(show_gui = False, use_egl = False)
    return PlayTableSimEnv(**env_kwargs)


def joint_word(joint, value):
    """The word of a joint position, by which half of its range it is in."""
    words = STATE_WORDS[joint]
    middle = (min(words.values()) + max(words.values())) / 2
    return max(words, key = words.get) if value > middle else min(words, key = words.get)


def read_seeds(path):
    """The last frame of each condition in a file, with its behavior and where it came from."""
    seeds = list()
    with h5py.File(path, "r") as file:
        for demo in sorted(file["data"].keys(), key = lambda name: int(name.split("_")[-1])):
            group = file["data"][demo]
            behavior = group.attrs.get("behavior", os.path.splitext(os.path.basename(path))[0])
            if behavior not in GOALS:
                raise ValueError(f"{path} {demo} is {behavior}, only {list(GOALS)} can be recombined")
            seeds.append({"behavior" : behavior, "source" : str(group.attrs.get("source", f"{path}:{demo}")),
                          "states" : group["obs/states"][-1].astype(np.float64), "proprio" : group["obs/proprio"][-1].astype(np.float64),
                          "action" : group["actions"][-1]})
    return seeds


class Simulator:
    """Puts a recorded frame into the simulator and measures how deep things are inside each other."""
    def __init__(self, env):
        self.env, self.p, self.cid = env, env.p, env.cid
        self.table = env.scene.doors[0].uid # drawer, door, switch, button and lights are links of the table
        self.robot = env.robot.robot_uid
        self.blocks = {name : obj.uid for name, obj in zip(BLOCK_INDEX, env.scene.movable_objects)}
        self.switch_threshold = env.scene.switches[0].trigger_threshold
        self.link_names = {-1 : "table"}
        for i in range(self.p.getNumJoints(self.table, physicsClientId = self.cid)):
            self.link_names[i] = self.p.getJointInfo(self.table, i, physicsClientId = self.cid)[12].decode()

    def set(self, states, proprio):
        states = states.copy()
        states[LIGHTBULB_INDEX] = float(states[STATE_INDEX["switch"]] > self.switch_threshold) # the light follows the switch
        # no physics step, so this is exactly the recorded frame. env.reset would step, and retry 1000 times if two blocks touch
        self.env.robot.reset(proprio)
        self.env.scene.reset(states, shuffle_movable_objects = False)

    def penetration(self, body_a, body_b):
        """How deep body_a is inside each link of body_b, in meters."""
        depth = dict()
        for point in self.p.getClosestPoints(body_a, body_b, distance = 0.0, physicsClientId = self.cid):
            depth[point[4]] = max(depth.get(point[4], 0.0), -point[8])
        return depth

    def contacts(self):
        """Depth per pair of bodies that matter: the robot in the table, and each block in the table and the robot."""
        result = {("robot", "table", link) : depth for link, depth in self.penetration(self.robot, self.table).items()}
        for name, uid in self.blocks.items():
            for other, other_uid in [("table", self.table), ("robot", self.robot)]:
                for link, depth in self.penetration(uid, other_uid).items():
                    result[(name, other, link)] = depth
        return result

    def new_contacts(self, before):
        """Pairs that are deeper inside each other than in the original frame."""
        return [key for key, depth in self.contacts().items() if depth > before.get(key, 0.0) + PENETRATION_TOLERANCE]

    def describe(self, key):
        name, other, link = key
        return f"{name} inside the {self.link_names.get(link, 'table') if other == 'table' else 'robot'}"

    def render(self):
        rgb, _ = self.env.get_camera_obs()
        return {"third_person" : rgb["rgb_static"], "eye_in_hand" : rgb["rgb_gripper"]}

    def robot_error(self, proprio):
        """How far the simulated robot is from the recorded one: tcp position and arm joints."""
        state = self.env.robot.get_observation()[0]
        return float(max(np.abs(state[0:3] - proprio[0:3]).max(), np.abs(state[7:14] - proprio[7:14]).max()))


def held_joints(seed, joints, radius):
    """The joints the gripper is holding or very close to, which must not move."""
    tcp = seed["proprio"][0:3]
    return [joint for joint in joints if np.linalg.norm(_anchor_position(ANCHOR[joint], seed["states"][None])[0] - tcp) < radius]


def recombine(sim, seed, combination, frozen):
    """
    Renders the seed with the joints of combination. Returns (images, states, notes), or (None, None, reason) if it isn't possible.
    - combination: {joint : word} for the joints that may change
    - frozen: joints that keep the position of the seed
    """
    for joint in frozen:
        if joint in combination and combination[joint] != joint_word(joint, seed["states"][STATE_INDEX[joint]]):
            return None, None, f"the gripper is at the {joint}"

    sim.set(seed["states"], seed["proprio"])
    before = sim.contacts()

    states = seed["states"].copy()
    for joint, word in combination.items():
        if joint not in frozen:
            states[STATE_INDEX[joint]] = STATE_WORDS[joint][word]
    drawer_shift = states[STATE_INDEX["drawer"]] - seed["states"][STATE_INDEX["drawer"]]
    for name, index in BLOCK_INDEX.items(): # blocks in the drawer move with it, the drawer opens towards -y
        x, z = states[index], states[index + 2]
        if DRAWER_BLOCK["x"][0] < x < DRAWER_BLOCK["x"][1] and z < DRAWER_BLOCK["z_below"]:
            states[index + 1] -= drawer_shift

    removed = list()
    for _ in range(len(BLOCK_INDEX) + 1): # remove the blocks in the way one round at a time, until nothing new is inside anything
        sim.set(states, seed["proprio"])
        problems = sim.new_contacts(before)
        if any(key[0] == "robot" for key in problems):
            return None, None, "the robot would be " + sim.describe(next(key for key in problems if key[0] == "robot"))[len("robot "):]
        blocks = sorted({key[0] for key in problems})
        if len(blocks) == 0:
            break
        for name in blocks:
            index = BLOCK_INDEX[name]
            states[index : index + 3], states[index + 3 : index + 6] = HIDDEN_POS[name], 0.0
            removed.append(f"{name}: " + sim.describe(next(key for key in problems if key[0] == name)))

    error = sim.robot_error(seed["proprio"])
    states[LIGHTBULB_INDEX] = float(states[STATE_INDEX["switch"]] > sim.switch_threshold)
    return sim.render(), states, {"removed_blocks" : removed, "robot_error" : round(error, 5)}


def file_name(behavior, combination):
    """Like drawer_closed__door_left__switch_off: the whole scene, the goal's joint in its end state."""
    joint, end = GOALS[behavior]
    words = STATE_WORDS[joint]
    full = {joint : max(words, key = words.get) if end == "high" else min(words, key = words.get), **combination}
    order = [name for name in ["drawer", "sliding_door", "switch", "green_light"] if name in full]
    return "__".join(f"{SHORT_NAMES.get(name, name)}_{full[name]}" for name in order)


def save_overview(path, rows, columns):
    """Every seed (row) in every combination (column), black where it was skipped."""
    size, header = 160, 54
    blank = np.zeros((size, size, 3), np.uint8)
    lines = [np.concatenate([cv2.resize(image, (size, size)) if image is not None else blank for image in row], axis = 1) for row in rows]
    top = np.full((header, size * len(columns), 3), 255, np.uint8)
    for i, column in enumerate(columns):
        for line, text in enumerate(column.split("__")):
            cv2.putText(top, text, (i * size + 4, 12 + 12 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, cv2.cvtColor(np.concatenate([top] + lines, axis = 0), cv2.COLOR_RGB2BGR))


def main(args):
    if os.path.isdir(args.output_folder) and len(os.listdir(args.output_folder)) > 0:
        sys.exit(f"Output folder {args.output_folder} is not empty, choose a new folder so nothing gets overwritten")
    unknown = [joint for joint in args.joints if joint not in STATE_INDEX]
    if len(unknown) > 0:
        sys.exit(f"Unknown joints {unknown}, choose from {list(STATE_INDEX)}")

    seeds_by_behavior = dict()
    for path in args.conditions:
        for seed in read_seeds(path):
            if args.hide_blocks: # before anything else, so the checks and the images both see the scene without blocks
                for name, index in BLOCK_INDEX.items():
                    seed["states"][index : index + 3], seed["states"][index + 3 : index + 6] = HIDDEN_POS[name], 0.0
            seeds_by_behavior.setdefault(seed["behavior"], list()).append(seed)

    sim = Simulator(make_env(args.env_config))
    manifest = dict()
    for behavior, seeds in seeds_by_behavior.items():
        goal_joint = GOALS[behavior][0]
        others = [joint for joint in args.joints if joint != goal_joint]
        combinations = [dict(zip(others, words)) for words in itertools.product(*[sorted(STATE_WORDS[joint]) for joint in others])]
        frozen = [[goal_joint] + [joint for joint in held_joints(seed, others, args.near_radius)] for seed in seeds]
        os.makedirs(os.path.join(args.output_folder, behavior))
        manifest[behavior] = dict()
        overview = [[None] * len(combinations) for _ in seeds]

        for column, combination in enumerate(combinations):
            name = file_name(behavior, combination)
            segments, entry = list(), {"conditions" : list(), "skipped" : list()}
            for row, seed in enumerate(seeds):
                images, states, notes = recombine(sim, seed, combination, frozen[row])
                if images is None:
                    entry["skipped"].append({"seed" : seed["source"], "reason" : notes})
                    continue
                overview[row][column] = images["third_person"]
                recording = Recording(seed["source"], states[None], seed["proprio"][None], seed["action"][None], ["third_person", "eye_in_hand"],
                                      lambda camera, frames, images = images: np.stack([images[camera] for _ in frames]))
                segments.append(Segment(recording, 0, 0, behavior))
                entry["conditions"].append({"seed" : seed["source"], "frozen" : frozen[row], **notes})
            if len(segments) > 0:
                write_conditions(os.path.join(args.output_folder, behavior, name + ".hdf5"), segments)
                save_preview(os.path.join(args.output_folder, behavior, name + ".png"), segments)
            manifest[behavior][name] = entry
            print(f"{behavior}/{name}: {len(segments)} of {len(seeds)}" + "".join(f"\n    skipped {s['seed']}: {s['reason']}" for s in entry["skipped"]))
        save_overview(os.path.join(args.output_folder, behavior, "overview.png"), overview, [file_name(behavior, c) for c in combinations])

    with open(os.path.join(args.output_folder, "manifest.json"), "w") as f:
        json.dump({"conditions" : args.conditions, "joints" : args.joints, "near_radius" : args.near_radius, "hide_blocks" : args.hide_blocks, "files" : manifest}, f, indent = 4)
    print(f"Saved to {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type = str, required = True, help = "new folder for the results")
    parser.add_argument("--conditions", type = str, nargs = "+", default = DEFAULT_CONDITIONS, help = "condition files to take the robot from")
    parser.add_argument("--joints", type = str, nargs = "+", default = list(STATE_INDEX), help = "joints to combine, the goal's own joint never changes")
    parser.add_argument("--near_radius", type = float, default = 0.1, help = "a joint whose handle is this close to the gripper, in meters, doesn't move")
    parser.add_argument("--hide_blocks", action = "store_true", help = "no blocks in any image, they are parked outside the camera view")
    parser.add_argument("--env_config", type = str, default = DEFAULT_ENV_CONFIG, help = "environment config")
    main(parser.parse_args())
