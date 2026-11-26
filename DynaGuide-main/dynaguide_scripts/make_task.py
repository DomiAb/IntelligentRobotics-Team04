import argparse
import copy
import itertools
import json
import os
import random
import sys

import h5py
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) # DynaGuide-main, for core

from core.calvin_utils import BLOCK_INDEX, PLACES, STATE_WORDS, generate_reset_state, parse_block, sample_block_poses
from core.goal_sequence import STEP_KEYS, validate_multigoal_config
from core.guidance_conditions import TABLE_THIRDS, save_preview
from recombine_conditions import DEFAULT_ENV_CONFIG, GOALS, STATE_INDEX, Simulator, joint_word, make_env
from scene_tools import (AREA_SPOT, JOINTS, block_area, held_joint_moved, hide_block, move_block, new_contacts, put_block_at_place, read_condition_poses,
                         save_rows, scene_name, write_rendered)

"""
Makes a task: the guidance conditions of every goal step and the experiment config, from a task file with the reset and the steps.

    python dynaguide_scripts/make_task.py --task calvin_exp_configs_examples/tasks/blue_table_to_open_drawer.json

Run it from DynaGuide-main. The output goes to datasets/guidance_conditions/<name of the task file>:
    <step>.hdf5 (.png)           the conditions of each step, <step>_contrast.hdf5 for a step with contrast_poses
    <step>_before.hdf5 (.png)    for a step that changes a joint: each condition once more, with the same robot but the joint still as it
                                 was before the step. The switch compares the two, see core/multigoal_guidance.py
    <step>_switch.hdf5 (.png)    for a release step into the drawer or onto the table: the block at a grid of places all over the area
                                 (GRID_SPACING), <step>_before.hdf5 is the same without the block. Only the switch uses these, the
                                 guidance doesn't. Its conditions are too few places: a block 4 cm from all of them looks like none.
                                 Other release steps: the conditions themselves and the same without the block
    <task>_all.png               the first conditions of every step, labeled with what the scene shows at that step
    experiment.json              the experiment config for run_dynaguide.py / run_experiments.py, its pos_examples point to the files above
    manifest.json                the scene of every step, every condition with the pose it came from, every skipped pose and why

The conditions come from the base sets of robot_poses: articulated/<behavior>/ (made with recombine_conditions.py --hide_blocks) and
blocks/<kind>/<side>/ (made with extract_block_poses.py), one file per scene. For each step the file of the scene the robot meets at that
step is taken, and every condition is rendered again with the exact joint values of the reset and the task's block where the story has
it. So the conditions show what the rollout shows, apart from the robot and what the step changes.

The task file:
    "env_setup"    the reset, every joint and every block fixed (a spot like table_right with "yaw", or "hidden")
    "reset_poses"  as in an experiment config (optional)
    "block"        the block the task works with, like "blue". Needed for block steps; the block poses are recolored to it
    "max"          at most this many conditions per step (optional, spread over the poses). The guidance sums over the conditions, so
                   steps with very different counts pull differently hard
    "avoid_behaviors", "on_avoid", "switch"   copied into the experiment config
    "random_scenes"  optional, {"scenes" : 4, "seed" : 0}: a task for a random scene, see make_random_task below. Then env_setup only fixes
                   what the task needs, every joint left out is random at each reset and every block left out lies somewhere on the table
    "steps"        the goal steps in order, each with
        "name"
        "poses"          the base set: "articulated/drawer_open", "blocks/grip_table/grip_right", "blocks/release_drawer", ...
        "scale", "alpha", "switch", "max_steps", ...   copied into the goal step of the experiment config
        "behaviors" / "where" / "any_behaviors"        what the env checks. Left out, it follows from the poses:
                         articulated/<behavior> -> that behavior; grip_* -> the block "held"; release_<place> -> the block at that place
                         and "released"
        "near"           only block poses whose block is this close (meters, x and y) to where the block is at that step
        "max"            at most this many conditions for this step
        "block_after"    where the block lies after a release step, a spot of calvin_utils.PLACES. Default: the center spot of the
                         area (drawer_center, table_right, cabinet_left_center, ...)
        "contrast_poses" a base set rendered in the scene before this step, as the switch's contrast. Steps after the first don't need
                         it, they contrast with the step before
        "home"           {"poses" : reset_poses file, "count" : 12, "steps" : 40, "scale" : 1.5, "alpha" : 30}: a home phase before the
                         step. <step>_home.hdf5 shows the scene before the step with the arm at count spawn poses of the file, and the
                         guidance pulls there for the first steps env steps of the step, see core/multigoal_guidance.py

The story, step by step:
- an articulated step moves its joint to the goal: its conditions show the pose's own joint value, later steps show the median of them
- a grip step needs the block resting in the area of its poses: its conditions show the block in the gripper, later steps show it held
- a release step shows the block where the pose let go of it, later steps show it at block_after. A block in the drawer moves with
  the drawer, so it is out of view once the drawer closes
- everything else stays as in the reset
A pose is skipped (listed in the manifest) where it doesn't fit: the gripper is at a joint the scene has elsewhere, the robot or a block
would be inside something, or a drawer pose's drawer is more than 2 cm off the scene's, which would move the block from under the gripper.
"""

POSES_ROOT = os.path.join("datasets", "robot_poses")
OUTPUT_ROOT = os.path.join("datasets", "guidance_conditions")
TASK_KEYS = {"description", "env_setup", "reset_poses", "block", "max", "avoid_behaviors", "on_avoid", "switch", "steps", "random_scenes"}
RANDOM_BLOCK_PLACE = "table" # where the blocks lie in a random scene, at rollouts and in the conditions
RANDOM_TRIES = 20 # random scenes tried per wanted scene, a scene where a step keeps no condition is drawn again
STEP_EXTRA_KEYS = {"poses", "near", "max", "block_after", "contrast_poses", "home"}
HOME_DEFAULTS = {"count" : 12, "steps" : 40, "scale" : 1.5, "alpha" : 30, "seed" : 0}
GOAL_KEYS = {"behaviors", "where", "any_behaviors"}
MAX_DRAWER_SHIFT = 0.02
GRID_SPACING = 0.03 # meters between the places of <step>_switch.hdf5
MIN_VISIBLE = 0.5 # a place where the image shows less than this part of the block (compared with the median place) is left out
DRAWER_GRID = ((0.04, 0.32), (-0.13, 0.02)) # x, and y with the drawer pushed back in: where blocks rest on the drawer floor
BUTTON_INDEX = 2 # the button joint in the scene state, the green light (STATE_INDEX) is what it toggles
HELD = "held"
BASE_COLOR = "blue"


class TaskError(ValueError):
    pass


def fixed_reset(env_setup):
    """The lowdim state of the reset. Raises an error unless every joint and block is fixed."""
    missing = [joint for joint in JOINTS if joint not in env_setup]
    if len(missing) > 0:
        raise TaskError(f"env_setup: fix every joint, {missing} missing")
    for name in BLOCK_INDEX:
        if name not in env_setup.get("blocks", dict()):
            raise TaskError(f"env_setup: fix every block, {name} missing (\"hidden\" takes it out of the scene)")
        low, high, _, yaw = parse_block(name, env_setup["blocks"][name])
        if not np.array_equal(low, high) or yaw is None:
            raise TaskError(f"env_setup: block {name} must be at a spot with a yaw, not in a region, so every reset is the same")
    state, _ = generate_reset_state(env_setup, rng = random.Random(0))
    return state


class PoseSet:
    """A base set of robot_poses: articulated/<behavior> or blocks/<kind>[/<action>_<side>]."""
    def __init__(self, name, root):
        self.name, self.folder = name, os.path.join(root, name)
        if not os.path.isdir(self.folder):
            raise TaskError(f"poses {name}: no folder {self.folder}")
        parts = name.split("/")
        self.kind = parts[0]
        if self.kind == "articulated":
            self.behavior = parts[1]
            if self.behavior not in GOALS:
                raise TaskError(f"poses {name}: {self.behavior} isn't an articulated behavior, choose from {list(GOALS)}")
            self.joint, end = GOALS[self.behavior]
            words = STATE_WORDS[self.joint]
            self.goal_word = max(words, key = words.get) if end == "high" else min(words, key = words.get)
        elif self.kind == "blocks":
            self.action, self.place = parts[1].split("_", 1) # grip / release, table / drawer / cabinet
            if self.action not in ["grip", "release"]:
                raise TaskError(f"poses {name}: expected blocks/grip_<place> or blocks/release_<place>")
            self.area = self.place + ("/" + parts[2].split("_", 1)[1] if len(parts) > 2 else "")
        else:
            raise TaskError(f"poses {name}: expected articulated/<behavior> or blocks/<kind>/...")

    def scene_file(self, state):
        """The file of the scene state, with the step's own joint at its goal."""
        target = state.copy()
        if self.kind == "articulated":
            target[STATE_INDEX[self.joint]] = STATE_WORDS[self.joint][self.goal_word]
        path = os.path.join(self.folder, scene_name(target) + ".hdf5")
        if not os.path.isfile(path):
            available = sorted(name[:-5] for name in os.listdir(self.folder) if name.endswith(".hdf5"))
            raise TaskError(f"poses {self.name}: no file for the scene {scene_name(target)} at this step. The set has {available}")
        return path


class Story:
    """The scene at each step: joint values, and where the task's block is (a resting position, or held)."""
    def __init__(self, reset, block):
        self.state, self.block = reset.copy(), block
        self.held = False

    def describe(self):
        words = ", ".join(f"{joint} {joint_word(joint, self.state[STATE_INDEX[joint]])}" for joint in JOINTS)
        if self.block is None:
            return words
        where = HELD if self.held else (block_area(self.block, self.state) or "hidden")
        return f"{words}, {self.block} {where}"

    def block_position(self):
        return self.state[BLOCK_INDEX[self.block] : BLOCK_INDEX[self.block] + 3]

    def set_joint(self, joint, value):
        """Moves a joint. A resting block in the drawer moves with it."""
        if joint == "drawer" and self.block is not None and not self.held and block_area(self.block, self.state) == "drawer":
            self.state[BLOCK_INDEX[self.block] + 1] -= value - self.state[STATE_INDEX["drawer"]]
        self.state[STATE_INDEX[joint]] = value


def default_goal(pose_set, block):
    if pose_set.kind == "articulated":
        return {"behaviors" : [pose_set.behavior]}
    if pose_set.action == "grip":
        return {"where" : {"blocks" : {block : HELD}}}
    word = pose_set.area.replace("/", "_") # table_right, drawer, cabinet_left
    return {"where" : {"blocks" : {block : [word, "released"]}}}


def render_step(sim, story, pose_set, entry, block, near = None):
    """The conditions of one step in the scene of story. Returns (rendered, attributes, kept, skipped)."""
    path = pose_set.scene_file(story.state)
    rendered, attributes, kept, skipped = list(), list(), list(), list()
    for pose in read_condition_poses(path):
        note = {"source" : str(pose["attrs"].get("source", pose["file"])), "file" : pose["file"]}
        state = story.state.copy()
        renamed = dict()
        if pose_set.kind == "articulated":
            value = pose["states"][STATE_INDEX[pose_set.joint]]
            if pose_set.joint == "drawer" and block is not None and not story.held and block_area(block, story.state) == "drawer":
                state[BLOCK_INDEX[block] + 1] -= value - state[STATE_INDEX["drawer"]] # the block lies in the drawer and moves with it
            state[STATE_INDEX[pose_set.joint]] = value
            if pose_set.joint == "green_light": # the pose presses the button down, the scene's button is released. The button
                state[BUTTON_INDEX] = pose["states"][BUTTON_INDEX] # springs back, so only this condition shows it pressed
        else:
            source = str(pose["attrs"].get("block", BASE_COLOR))
            position = pose["states"][BLOCK_INDEX[source] : BLOCK_INDEX[source] + 3]
            if near is not None and np.linalg.norm(position[:2] - story.block_position()[:2]) > near:
                continue
            if pose_set.place == "drawer":
                shift = state[STATE_INDEX["drawer"]] - pose["states"][STATE_INDEX["drawer"]]
                if abs(shift) > MAX_DRAWER_SHIFT:
                    skipped.append({**note, "reason" : f"its drawer is {-shift:+.3f} off the scene's, the block would leave the gripper"})
                    continue
            move_block(state, pose["states"], source, block, pose["proprio"][3:6])
            if pose_set.place == "drawer": # the block lies in the drawer of the pose, the scene's drawer differs by up to 2 cm
                state[BLOCK_INDEX[block] + 1] -= state[STATE_INDEX["drawer"]] - pose["states"][STATE_INDEX["drawer"]]
            renamed = {block : source}
        state[4] = float(state[STATE_INDEX["switch"]] > sim.switch_threshold) # the lightbulb follows the switch
        joint = held_joint_moved(pose["states"], pose["proprio"][0:3], state)
        problem = f"the gripper is at the {joint}, which is on the other side in this scene" if joint is not None else \
            new_contacts(sim, pose["states"], state, pose["proprio"], renamed) # the task's block compared with the pose's block
        if problem is not None:
            skipped.append({**note, "reason" : problem})
            continue
        rendered.append((sim.render(), state, pose))
        attributes.append({"pose_file" : pose["file"]})
        kept.append({**note, "drawer" : round(float(state[STATE_INDEX["drawer"]]), 4),
                     **({block : [round(float(x), 4) for x in state[BLOCK_INDEX[block] : BLOCK_INDEX[block] + 3]]} if block else {})})
    limit = entry.get("max")
    if limit is not None and len(rendered) > limit:
        chosen = np.linspace(0, len(rendered) - 1, limit).round().astype(int)
        rendered, attributes, kept = [rendered[i] for i in chosen], [attributes[i] for i in chosen], [kept[i] for i in chosen]
    return path, rendered, attributes, kept, skipped


def before_state(story, pose_set, state, block):
    """The scene of a condition with the joint of the step back where it was before the step, everything else as in the condition."""
    before = state.copy()
    value = story.state[STATE_INDEX[pose_set.joint]]
    if pose_set.joint == "drawer" and block is not None and not story.held and block_area(block, story.state) == "drawer":
        before[BLOCK_INDEX[block] + 1] -= value - state[STATE_INDEX["drawer"]] # the block in the drawer moves with it
    before[STATE_INDEX[pose_set.joint]] = value
    if pose_set.joint == "green_light": # the button isn't pressed yet either
        before[BUTTON_INDEX] = 0.0
    return before


def render_before(sim, story, pose_set, rendered, block):
    """
    The twin of every condition of an articulated step, see before_state, or of a release step, the block gone. The robot may be inside something here, like the gripper in
    the closed drawer. That doesn't matter: the image is only compared with its condition, to see what the step changes.
    """
    twins = list()
    for _, state, pose in rendered:
        if pose_set.kind == "articulated":
            before = before_state(story, pose_set, state, block)
        else:
            before = state.copy()
            hide_block(before, block)
        sim.set(before, pose["proprio"])
        before[4] = float(before[STATE_INDEX["switch"]] > sim.switch_threshold) # the lightbulb follows the switch
        twins.append((sim.render(), before, pose))
    return twins


def render_home(sim, story, spec):
    """The scene of the story with the arm at spawn poses: the home examples of a step."""
    with open(spec["poses"]) as f:
        robots = json.load(f)["robot_states"]
    chosen = random.Random(spec["seed"]).sample(range(len(robots)), min(spec["count"], len(robots)))
    rendered = list()
    for number in chosen:
        state, proprio = story.state.copy(), np.array(robots[number], dtype = np.float64)
        state[4] = float(state[STATE_INDEX["switch"]] > sim.switch_threshold) # the lightbulb follows the switch
        sim.set(state, proprio)
        pose = {"attrs" : {"source" : f"{spec['poses']}:{number}"}, "file" : spec["poses"], "proprio" : proprio, "action" : np.zeros(7)}
        rendered.append((sim.render(), state, pose))
    return rendered


def place_grid(pose_set, drawer):
    """(x, y) places all over the area of a release step, None for an area without a grid (the cabinet). drawer: how far it is open."""
    if pose_set.place == "drawer":
        (x_low, x_high), (y_low, y_high) = DRAWER_GRID
        y_low, y_high = y_low - drawer, y_high - drawer
    elif pose_set.place == "table":
        low, high, _ = PLACES["table"]
        third = {"left" : 0, "center" : 1, "right" : 2}.get(pose_set.area.split("/")[-1])
        edges = [low[0], *TABLE_THIRDS, high[0]]
        x_low, x_high = (low[0], high[0]) if third is None else (edges[third], edges[third + 1])
        y_low, y_high = low[1], high[1]
    else:
        return None
    axis = lambda start, end: np.arange(start, end + 1e-6, GRID_SPACING)
    return list(itertools.product(axis(x_low, x_high), axis(y_low, y_high)))


def render_grid(sim, pose_set, rendered, block):
    """
    The images the switch of a release step compares: (with the block, without it), the block at each place of the grid, the robot of
    one of the conditions, far from the place so it hides little (one of the four farthest). None for an area without a grid.
    A place where the block is hidden (by the table or the arm) is left out: such a pair has no change to compare.
    """
    places = place_grid(pose_set, float(rendered[0][1][STATE_INDEX["drawer"]]))
    if places is None:
        return None
    height, index = float(np.median([state[BLOCK_INDEX[block] + 2] for _, state, _ in rendered])), BLOCK_INDEX[block]
    goals, befores = list(), list()
    for number, (x, y) in enumerate(places):
        farthest = sorted(range(len(rendered)), key = lambda k: -np.hypot(rendered[k][2]["proprio"][0] - x, rendered[k][2]["proprio"][1] - y))[:4]
        _, state, pose = rendered[farthest[number % len(farthest)]]
        goal = state.copy()
        goal[index : index + 3] = [x, y, height]
        before = goal.copy()
        hide_block(before, block)
        for images, scene in [(goals, goal), (befores, before)]:
            sim.set(scene, pose["proprio"])
            images.append((sim.render(), scene, pose))
    changed = lambda a, b: int((np.abs(a["third_person"].astype(int) - b["third_person"].astype(int)).max(axis = 2) > 20).sum())
    pixels = np.array([changed(goal[0], before[0]) for goal, before in zip(goals, befores)])
    visible = np.flatnonzero(pixels >= MIN_VISIBLE * np.median(pixels))
    return [goals[k] for k in visible], [befores[k] for k in visible]


def check_story(story, rendered, pose_set, block, reset):
    """Every condition shows the story's scene, apart from the step's own joint and the task's block. Raises an error otherwise (a bug)."""
    for _, state, pose in rendered:
        for joint in JOINTS:
            if pose_set.kind == "articulated" and joint == pose_set.joint:
                continue
            if abs(state[STATE_INDEX[joint]] - story.state[STATE_INDEX[joint]]) > 1e-9:
                raise AssertionError(f"{pose['file']}: {joint} differs from the scene of this step")
        for name, index in BLOCK_INDEX.items():
            if name != block and not np.allclose(state[index : index + 6], reset[index : index + 6]):
                raise AssertionError(f"{pose['file']}: block {name} differs from the reset")
        if block is not None and pose_set.kind == "articulated": # the block rests where the story has it, in the drawer relative to the drawer
            index, drawer = BLOCK_INDEX[block], STATE_INDEX["drawer"]
            expected = story.state[index : index + 3].copy()
            if block_area(block, story.state) == "drawer":
                expected[1] -= state[drawer] - story.state[drawer]
            if not np.allclose(state[index : index + 3], expected):
                raise AssertionError(f"{pose['file']}: the {block} block isn't where the story has it")


def advance(story, pose_set, rendered, entry):
    """The scene after the step."""
    if pose_set.kind == "articulated":
        values = [state[STATE_INDEX[pose_set.joint]] for _, state, _ in rendered]
        story.set_joint(pose_set.joint, float(np.median(values)) if pose_set.joint != "green_light" else STATE_WORDS["green_light"][pose_set.goal_word])
    elif pose_set.action == "grip":
        story.held = True
    else:
        place = entry.get("block_after", AREA_SPOT[pose_set.area])
        put_block_at_place(story.state, story.block, place)
        story.held = False


def check_before(story, pose_set, name):
    if pose_set.kind == "articulated":
        if story.held:
            raise TaskError(f"step {name}: an articulated step while the block is held isn't supported, release it first")
        return
    if story.block is None:
        raise TaskError(f"step {name}: a block step needs \"block\" in the task")
    if pose_set.action == "grip":
        if story.held:
            raise TaskError(f"step {name}: the block is already held")
        area = block_area(story.block, story.state)
        if area is None or not area.startswith(pose_set.area):
            raise TaskError(f"step {name}: the {story.block} block is at {area} at this step, the poses {pose_set.name} grip it at {pose_set.area}")
    if pose_set.place == "drawer" and joint_word("drawer", story.state[STATE_INDEX["drawer"]]) != "open":
        raise TaskError(f"step {name}: the poses {pose_set.name} need the drawer open at this step")


def make_task(task, task_name, sim, root = POSES_ROOT, output = None):
    unknown = set(task) - TASK_KEYS
    if len(unknown) > 0:
        raise TaskError(f"unknown keys {sorted(unknown)} in the task, choose from {sorted(TASK_KEYS)}")
    block = task.get("block")
    if block is not None and block not in BLOCK_INDEX:
        raise TaskError(f"block must be one of {list(BLOCK_INDEX)}")
    reset = fixed_reset(task["env_setup"])
    story = Story(reset, block)
    output = output or os.path.join(OUTPUT_ROOT, task_name)
    if os.path.isdir(output) and len(os.listdir(output)) > 0:
        raise TaskError(f"{output} is not empty. Delete it to make the task again")
    os.makedirs(output, exist_ok = True)

    config = {"description" : task.get("description", ""), "env_setup" : task["env_setup"]}
    for key in ["reset_poses", "avoid_behaviors", "on_avoid", "switch"]:
        if key in task:
            config[key] = task[key]
    config["goal_sequence"] = list()
    manifest = {"task" : task, "steps" : list()}
    rows = list()
    for number, entry in enumerate(task["steps"]):
        name = entry.get("name", f"step_{number}")
        unknown = set(entry) - STEP_EXTRA_KEYS - set(STEP_KEYS)
        if len(unknown) > 0:
            raise TaskError(f"step {name}: unknown keys {sorted(unknown)}")
        entry = {**entry, "max" : entry.get("max", task.get("max"))}
        pose_set = PoseSet(entry["poses"], root)
        check_before(story, pose_set, name)
        scene = story.describe()
        step = {key : value for key, value in entry.items() if key not in STEP_EXTRA_KEYS}
        step["name"] = name
        if not GOAL_KEYS & set(entry):
            step.update(default_goal(pose_set, block))

        if "contrast_poses" in entry:
            contrast_set = PoseSet(entry["contrast_poses"], root)
            path, rendered, attributes, kept, skipped = render_step(sim, story, contrast_set, entry, block, entry.get("near"))
            check_story(story, rendered, contrast_set, block, reset)
            if len(rendered) == 0:
                raise TaskError(f"step {name}: no contrast condition fits, see the skipped poses in {output}/manifest.json")
            contrast_file = os.path.join(output, f"{name}_contrast.hdf5")
            segments = write_rendered(contrast_file, rendered, f"{name}_contrast", attributes)
            save_preview(contrast_file[:-5] + ".png", segments)
            rows.append((f"{name} contrast | {scene}", [images["third_person"] for images, _, _ in rendered]))
            step["switch"] = {**step.get("switch", dict()), "contrast_examples" : contrast_file}
            manifest["steps"].append({"name" : f"{name}_contrast", "poses" : entry["contrast_poses"], "base_file" : path, "scene" : scene,
                                      "conditions" : kept, "skipped" : skipped})

        path, rendered, attributes, kept, skipped = render_step(sim, story, pose_set, entry, block, entry.get("near"))
        check_story(story, rendered, pose_set, block, reset)
        print(f"{name:32s} {len(rendered):3d} conditions, {len(skipped)} skipped | {scene}")
        manifest["steps"].append({"name" : name, "poses" : entry["poses"], "base_file" : path, "scene" : scene,
                                  "conditions" : kept, "skipped" : skipped})
        if len(rendered) == 0:
            reasons = "\n".join(f"    {entry['file']}: {entry['reason']}" for entry in skipped)
            raise TaskError(f"step {name}: no condition of {path} fits the scene {scene}. The skipped poses:\n{reasons}")
        condition_file = os.path.join(output, f"{name}.hdf5")
        segments = write_rendered(condition_file, rendered, name, attributes)
        save_preview(condition_file[:-5] + ".png", segments)
        rows.append((f"{name} | {scene}", [images["third_person"] for images, _, _ in rendered]))
        step["pos_examples"] = condition_file
        if pose_set.kind == "articulated" or pose_set.action == "release": # a grip changes nothing that stays in place, its switch is the older one
            grid = render_grid(sim, pose_set, rendered, block) if pose_set.kind == "blocks" else None
            if grid is None:
                twins, twin_attributes = render_before(sim, story, pose_set, rendered, block), attributes
            else:
                goals, twins = grid
                twin_attributes = [{"pose_file" : pose["file"]} for _, _, pose in goals]
                switch_file = os.path.join(output, f"{name}_switch.hdf5")
                segments = write_rendered(switch_file, goals, f"{name}_switch", twin_attributes)
                save_preview(switch_file[:-5] + ".png", segments)
                rows.append((f"{name} switch, {len(goals)} places | {scene}", [images["third_person"] for images, _, _ in goals]))
                step["switch"] = {**step.get("switch", dict()), "goal_examples" : switch_file}
            before_file = os.path.join(output, f"{name}_before.hdf5")
            segments = write_rendered(before_file, twins, f"{name}_before", twin_attributes)
            save_preview(before_file[:-5] + ".png", segments)
            rows.append((f"{name} before | {scene}", [images["third_person"] for images, _, _ in twins]))
            step["switch"] = {**step.get("switch", dict()), "before_examples" : before_file}
            if pose_set.kind == "blocks": # the block can lie anywhere in the area, only the patches criterion can deal with that
                step["switch"].setdefault("default_criterion", "patches")
        if "home" in entry: # rendered in the scene before the step
            spec = {**HOME_DEFAULTS, **entry["home"]}
            home = render_home(sim, story, spec)
            home_file = os.path.join(output, f"{name}_home.hdf5")
            segments = write_rendered(home_file, home, f"{name}_home")
            save_preview(home_file[:-5] + ".png", segments)
            rows.append((f"{name} home | {scene}", [images["third_person"] for images, _, _ in home]))
            step["home"] = {"examples" : home_file, "steps" : spec["steps"], "scale" : spec["scale"], "alpha" : spec["alpha"]}
        config["goal_sequence"].append(step)
        advance(story, pose_set, rendered, entry)

    manifest["final_scene"] = story.describe()
    validate_multigoal_config(copy.deepcopy(config))
    with open(os.path.join(output, "experiment.json"), "w") as f:
        json.dump(config, f, indent = 4)
    with open(os.path.join(output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent = 4)
    save_rows(os.path.join(output, f"{os.path.basename(os.path.normpath(output))}_all.png"), rows)
    print(f"After the last step: {story.describe()}")
    print(f"Saved to {output}: the conditions, {os.path.basename(os.path.normpath(output))}_all.png and experiment.json")
    return config


def random_env_setup(task, rng):
    """
    One random scene as a fixed env_setup: the task's joints, the others at a random end, the blocks env_setup fixes (the task's block)
    where it has them, every other block at a random place on the table.
    """
    fixed_blocks = task["env_setup"].get("blocks", dict())
    env_setup = {key : value for key, value in task["env_setup"].items() if key != "blocks"}
    for joint in JOINTS:
        if joint not in env_setup:
            env_setup[joint] = rng.choice(sorted(STATE_WORDS[joint]))
    drawer = STATE_WORDS["drawer"][env_setup["drawer"]] if isinstance(env_setup["drawer"], str) else env_setup["drawer"]
    poses = sample_block_poses({color : fixed_blocks.get(color, RANDOM_BLOCK_PLACE) for color in BLOCK_INDEX}, drawer, rng = rng)
    env_setup["blocks"] = {color : fixed_blocks[color] if color in fixed_blocks else
                           {"pos" : [round(float(x), 5) for x in poses[index - 6 : index - 3]], "yaw" : round(float(poses[index - 1]), 5)}
                           for color, index in BLOCK_INDEX.items()}
    return env_setup


def merge_conditions(paths, output):
    """Writes the conditions of several files into one, in the order given, so condition k of a merged file and its twin stay pairs."""
    with h5py.File(output, "w") as merged:
        data, number = merged.create_group("data"), 0
        for path in paths:
            with h5py.File(path, "r") as source:
                for key, value in source["data"].attrs.items():
                    if key != "total":
                        data.attrs[key] = value
                for demo in sorted(source["data"].keys(), key = lambda name: int(name.split("_")[-1])):
                    source.copy(source["data"][demo], data, name = f"demo_{number}")
                    number += 1
        data.attrs["total"] = sum(int(data[demo].attrs.get("num_samples", 1)) for demo in data.keys())


def make_random_task(task, task_name, sim, root = POSES_ROOT, output = None):
    """
    A task for random scenes: the joints env_setup leaves out are random at each reset, and every block lies at a random place on the
    table. The conditions show such scenes too: the task is made scenes times (random_scenes), each time in its own random scene, with
    max / scenes conditions per step, and the files of each step are merged. Each scene is in scene_<n>/ with its manifest and previews.
    The switch's twins are rendered in the scene of their condition, so the pairs only differ by the step's change. A grip or release
    step needs its block at a fixed place: give it as a spot with a yaw in env_setup (the task's "block"), the others are random.
    """
    settings = task["random_scenes"]
    scenes, seed = settings.get("scenes", 4), settings.get("seed", 0)
    for color, spec in task["env_setup"].get("blocks", dict()).items():
        low, high, _, yaw = parse_block(color, spec)
        if not np.array_equal(low, high) or yaw is None:
            raise TaskError(f"random_scenes: block {color} in env_setup must be at a spot with a yaw (the task's block), leave the others out")
    output = output or os.path.join(OUTPUT_ROOT, task_name)
    if os.path.isdir(output) and len(os.listdir(output)) > 0:
        raise TaskError(f"{output} is not empty. Delete it to make the task again")
    per_scene = int(np.ceil(task.get("max", 12) / scenes))
    rng, configs, setups = random.Random(seed), list(), list()
    for number in range(scenes):
        for attempt in range(RANDOM_TRIES):
            env_setup = random_env_setup(task, rng)
            scene_task = {key : value for key, value in task.items() if key != "random_scenes"}
            scene_task.update(env_setup = env_setup, max = per_scene)
            folder = os.path.join(output, f"scene_{number}")
            try:
                configs.append(make_task(scene_task, f"{task_name}_scene_{number}", sim, root, folder))
                setups.append(env_setup)
                break
            except TaskError as error:
                if "fits" not in str(error): # "no condition fits", "no contrast condition fits"
                    raise
                print(f"scene {number}: {str(error).splitlines()[0]} Drawing another scene")
                for name in os.listdir(folder):
                    os.remove(os.path.join(folder, name))
        else:
            raise TaskError(f"no random scene in {RANDOM_TRIES} tries where every step keeps a condition")

    config = copy.deepcopy(configs[0])
    config["description"] = task.get("description", "")
    fixed_blocks = task["env_setup"].get("blocks", dict())
    config["env_setup"] = {**task["env_setup"], "blocks" : {color : fixed_blocks.get(color, RANDOM_BLOCK_PLACE) for color in BLOCK_INDEX}}
    for number, step in enumerate(config["goal_sequence"]):
        name = step["name"]
        step["pos_examples"] = os.path.join(output, f"{name}.hdf5")
        merge_conditions([c["goal_sequence"][number]["pos_examples"] for c in configs], step["pos_examples"])
        if "home" in step:
            step["home"]["examples"] = os.path.join(output, os.path.basename(configs[0]["goal_sequence"][number]["home"]["examples"]))
            merge_conditions([c["goal_sequence"][number]["home"]["examples"] for c in configs], step["home"]["examples"])
        for key in ["contrast_examples", "before_examples", "goal_examples"]:
            if key in step.get("switch", dict()):
                step["switch"][key] = os.path.join(output, os.path.basename(configs[0]["goal_sequence"][number]["switch"][key]))
                merge_conditions([c["goal_sequence"][number]["switch"][key] for c in configs], step["switch"][key])
        with h5py.File(step["pos_examples"], "r") as f:
            print(f"{name:32s} {len(f['data'])} conditions over {scenes} random scenes")
    validate_multigoal_config(copy.deepcopy(config))
    with open(os.path.join(output, "experiment.json"), "w") as f:
        json.dump(config, f, indent = 4)
    with open(os.path.join(output, "manifest.json"), "w") as f:
        json.dump({"task" : task, "scenes" : setups}, f, indent = 4)
    print(f"Saved to {output}: the merged conditions and experiment.json, each scene in scene_<n>/")
    return config


def main(args):
    with open(args.task, "r") as f:
        task = json.load(f)
    name = os.path.splitext(os.path.basename(args.task))[0]
    sim = Simulator(make_env(args.env_config))
    try:
        if "random_scenes" in task:
            make_random_task(task, name, sim, args.poses_root, args.output_folder)
        else:
            make_task(task, name, sim, args.poses_root, args.output_folder)
    except TaskError as error:
        sys.exit(f"{args.task}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type = str, required = True, help = "the task file, see the top of this file")
    parser.add_argument("--output_folder", type = str, default = None, help = "default: datasets/guidance_conditions/<name of the task file>")
    parser.add_argument("--poses_root", type = str, default = POSES_ROOT, help = "the robot_poses collection")
    parser.add_argument("--env_config", type = str, default = DEFAULT_ENV_CONFIG, help = "environment config")
    main(parser.parse_args())
