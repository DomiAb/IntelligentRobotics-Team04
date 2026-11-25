"""
Tests of the block base sets and the tasks (dynaguide_scripts/extract_block_poses.py, scene_tools.py, make_task.py). Run from DynaGuide-main:

    python -m pytest tests/test_scene_conditions.py -v

Needs the calvin_env package (PYTHONPATH with calvin/calvin_env), the scripts import the simulator. No simulator is started. The last
tests check the made tasks in datasets/guidance_conditions and are skipped until make_task.py has made them.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "dynaguide_scripts"))
os.chdir(REPO)

pytest.importorskip("calvin_env")

from core.calvin_utils import BLOCK_INDEX, HIDDEN_POS, PLACES
from core.goal_sequence import SequentialGoalTracker, validate_multigoal_config
from extract_block_poses import base_set_folder, candidate_frames, frame_kinds
from make_task import PoseSet, Story, TaskError, before_state, check_before, default_goal, fixed_reset
from scene_tools import block_area, held_joint_moved, joint_combinations, long_side_along_gripper, put_block_at_place, scene_name

TASKS = "calvin_exp_configs_examples/tasks"
CHAINS = ["drawer_button_door", "door_drawer_switch", "button_switch_drawer", "door_switch_button"] # the tasks of run_goal_switch.bat
TASK_NAMES = ["blue_table_to_open_drawer", "blue_table_to_open_drawer_one_step", "single_block_drawer"]


def load(path):
    with open(path) as f:
        return json.load(f)


def proprio(tcp, width = 0.08, gripper = 1.0):
    p = np.zeros(15)
    p[0:3], p[6], p[14] = tcp, width, gripper
    return p


def recording(blue, tcp, width, gripper, drawer = 0.18):
    """A short recording with only the blue block: lists of block positions, tcp positions, gripper widths and gripper actions."""
    states = np.zeros((len(blue), 24))
    states[:, 1] = drawer
    for name, index in BLOCK_INDEX.items():
        states[:, index : index + 3] = HIDDEN_POS[name]
    states[:, 12:15] = blue
    prop = np.stack([proprio(t, width = w, gripper = g) for t, w, g in zip(tcp, width, gripper)])
    return states, prop

# --- finding the poses -------------------------------------------------------------------------------------------------------------

def test_grip_on_table_then_release_in_drawer():
    on_table, lifted, in_drawer = [0.17, -0.09, 0.46], [0.17, -0.09, 0.55], [0.17, -0.25, 0.3624]
    blue = [on_table, on_table, on_table, lifted, in_drawer, in_drawer]
    tcp = [[0.17, -0.09, 0.50], [0.17, -0.09, 0.47], [0.17, -0.09, 0.47], [0.17, -0.09, 0.56], [0.17, -0.25, 0.40], [0.17, -0.25, 0.42]]
    width = [0.08, 0.04, 0.04, 0.04, 0.04, 0.078]
    gripper = [1, -1, -1, -1, -1, 1]
    kinds = frame_kinds(*recording(blue, tcp, width, gripper))
    assert list(kinds[("grip_table", "blue")]) == [False, True, True, False, False, False] # lifted 9 cm, or held in the drawer, is no grip on the table
    assert list(kinds[("release_drawer", "blue")]) == [False, False, False, False, False, True]
    assert not kinds[("release_table", "blue")].any() and not kinds[("grip_drawer", "blue")].any()
    kinds = frame_kinds(*recording(blue, tcp, width, gripper, drawer = 0.12)) # the drawer not open as far as COMPLETED
    assert not kinds[("release_drawer", "blue")].any()


def test_a_gripper_closed_next_to_a_block_is_no_grip():
    on_table = [0.17, -0.09, 0.46]
    kinds = frame_kinds(*recording([on_table] * 3, [[0.17, -0.09, 0.50], [0.20, -0.09, 0.47], [0.20, -0.09, 0.47]],
                                   [0.08, 0.0, 0.0], [1, -1, -1]))
    assert not kinds[("grip_table", "blue")].any()


def test_candidate_frames():
    mask = np.zeros(30, dtype = bool)
    mask[2:20] = True
    mask[25] = True
    assert candidate_frames(mask, 8) == [3, 11, 19, 25]


def test_areas_and_base_set_folders():
    state = np.zeros(24)
    for position, area in [([0.05, -0.09, 0.46], "table/left"), ([0.175, -0.09, 0.46], "table/center"), ([0.30, -0.12, 0.46], "table/right"),
                           ([0.18, -0.25, 0.3624], "drawer"), ([-0.24, 0.105, 0.461], "cabinet/left"), ([0.04, 0.105, 0.461], "cabinet/right"),
                           (HIDDEN_POS["blue"], None)]:
        state[12:15] = position
        assert block_area("blue", state) == area
    assert base_set_folder("blocks", "grip_cabinet", "cabinet/left") == os.path.join("blocks", "grip_cabinet", "grip_left")
    assert base_set_folder("blocks", "release_drawer", "drawer") == os.path.join("blocks", "release_drawer")

# --- scenes and blocks -----------------------------------------------------------------------------------------------------------------

def test_scene_names_and_combinations():
    state = np.zeros(24)
    state[[0, 1, 3, 5]] = [0.27, 0.2, 0.0, 1.0]
    assert scene_name(state) == "drawer_open__door_left__switch_off__green_light_on"
    assert len(joint_combinations()) == 16 and len(joint_combinations(["drawer"])) == 8
    assert all("drawer" not in combination for combination in joint_combinations(["drawer"]))


def test_block_at_a_drawer_place_follows_the_drawer():
    state = np.zeros(24)
    state[1] = 0.2
    put_block_at_place(state, "blue", "drawer_center")
    assert np.allclose(state[12:15], [0.18, -0.085 - 0.2, PLACES["drawer_center"].low[2]])


def test_recolored_block_turns_its_long_side_along_the_gripper():
    from scipy.spatial.transform import Rotation
    gripper = [np.pi, 0.0, 0.0] # pointing down, its x axis along the world's x
    cube = [0.0, np.pi / 2, 0.0] # its local x axis points down, its local z axis along the world's x: a pink block would stand upright
    turned = Rotation.from_euler("xyz", long_side_along_gripper(cube, gripper)).as_matrix()
    assert abs(turned[:, 0] @ Rotation.from_euler("xyz", gripper).as_matrix()[:, 0]) > 0.99
    assert np.isclose(np.linalg.det(turned), 1.0)


def test_held_joint_is_not_moved():
    pose_states = np.zeros(24) # closed drawer, gripper at its handle
    moved = pose_states.copy()
    moved[1] = 0.18
    assert held_joint_moved(pose_states, np.array([0.175, -0.205, 0.40]), moved) == "drawer"
    assert held_joint_moved(pose_states, np.array([0.175, -0.205, 0.40]), pose_states) is None

# --- the task's story ---------------------------------------------------------------------------------------------------------------

def test_tasks_have_a_fixed_reset():
    for name in TASK_NAMES:
        reset = fixed_reset(load(f"{TASKS}/{name}.json")["env_setup"])
        assert np.allclose(reset[12:15], np.array(PLACES["table_right"].low) + [0.0, -0.03, 0.0])
        assert np.allclose(reset[6:9], HIDDEN_POS["red"]) and np.allclose(reset[18:21], HIDDEN_POS["pink"])
    env_setup = dict(load(f"{TASKS}/single_block_drawer.json")["env_setup"])
    with pytest.raises(TaskError):
        fixed_reset({**env_setup, "blocks" : {**env_setup["blocks"], "blue" : "table"}}) # a region, every reset would differ
    with pytest.raises(TaskError):
        fixed_reset({key : value for key, value in env_setup.items() if key != "switch"})


def test_story_moves_a_block_in_the_drawer_with_it():
    story = Story(fixed_reset(load(f"{TASKS}/single_block_drawer.json")["env_setup"]), "blue")
    story.set_joint("drawer", 0.2)
    assert np.allclose(story.state[12:15], np.array(PLACES["table_right"].low) + [0.0, -0.03, 0.0]) # on the table, stays
    put_block_at_place(story.state, "blue", "drawer_center")
    story.set_joint("drawer", 0.0)
    assert np.allclose(story.state[12:15], [0.18, -0.085, PLACES["drawer_center"].low[2]]) and block_area("blue", story.state) == "drawer"


def test_steps_must_fit_the_story(tmp_path):
    for folder in ["articulated/drawer_open", "blocks/grip_table/grip_left", "blocks/grip_table/grip_right", "blocks/release_drawer"]:
        (tmp_path / folder).mkdir(parents = True)
    story = Story(fixed_reset(load(f"{TASKS}/single_block_drawer.json")["env_setup"]), "blue") # drawer closed, blue on the right of the table
    with pytest.raises(TaskError, match = "grip it at table/left"):
        check_before(story, PoseSet("blocks/grip_table/grip_left", str(tmp_path)), "grip")
    with pytest.raises(TaskError, match = "drawer open"):
        check_before(story, PoseSet("blocks/release_drawer", str(tmp_path)), "release")
    check_before(story, PoseSet("blocks/grip_table/grip_right", str(tmp_path)), "grip")
    story.held = True
    with pytest.raises(TaskError, match = "held"):
        check_before(story, PoseSet("articulated/drawer_open", str(tmp_path)), "open")
    assert default_goal(PoseSet("articulated/drawer_open", str(tmp_path)), "blue") == {"behaviors" : ["drawer_open"]}
    assert default_goal(PoseSet("blocks/grip_table/grip_right", str(tmp_path)), "blue") == {"where" : {"blocks" : {"blue" : "held"}}}
    assert default_goal(PoseSet("blocks/release_drawer", str(tmp_path)), "blue") == {"where" : {"blocks" : {"blue" : ["drawer", "released"]}}}

def test_before_state(tmp_path):
    (tmp_path / "articulated" / "drawer_close").mkdir(parents = True)
    (tmp_path / "articulated" / "button_on").mkdir(parents = True)
    story = Story(fixed_reset(load(f"{TASKS}/single_block_drawer.json")["env_setup"]), "blue")
    story.set_joint("drawer", 0.2)
    put_block_at_place(story.state, "blue", "drawer_center")
    closed = story.state.copy() # what a drawer_close condition shows: closed, the block in it
    closed[1], closed[13] = 0.0, closed[13] + 0.2
    before = before_state(story, PoseSet("articulated/drawer_close", str(tmp_path)), closed, "blue")
    assert np.allclose(before, story.state) # open again, the block moved out with it
    pressed = story.state.copy()
    pressed[2], pressed[5] = 0.02, 1.0
    assert np.allclose(before_state(story, PoseSet("articulated/button_on", str(tmp_path)), pressed, "blue"), story.state)


def test_chains_test_every_joint_behavior():
    """Every joint behavior is a step that the switch has to end, so not the last one of a chain."""
    switched = set()
    for name in CHAINS:
        task = load(f"{TASKS}/{name}.json")
        fixed_reset(task["env_setup"])
        switched |= {step["poses"].split("/")[1] for step in task["steps"][:-1]}
    assert switched == {"drawer_open", "drawer_close", "door_left", "door_right", "switch_on", "switch_off", "button_on", "button_off"}

# --- the made tasks -----------------------------------------------------------------------------------------------------------------

def made(name):
    folder = os.path.join("datasets", "guidance_conditions", name)
    if not os.path.isfile(os.path.join(folder, "experiment.json")):
        pytest.skip(f"make the task {name} first (make_task.py)")
    return folder


@pytest.mark.parametrize("name", TASK_NAMES)
def test_made_task_follows_its_story(name):
    import h5py
    from core.guidance_conditions import StateView, find_frames
    folder = made(name)
    config, manifest = load(os.path.join(folder, "experiment.json")), load(os.path.join(folder, "manifest.json"))
    validate_multigoal_config(config)
    reset = fixed_reset(config["env_setup"])
    on_table = lambda s: np.allclose(s[12:15], reset[12:15], atol = 1e-6)
    blue = lambda s, p, spec: bool(find_frames(StateView(s, p), {"blocks" : {"blue" : spec}})[0])
    expected = {"open_drawer_contrast" : lambda s, p: on_table(s) and s[1] < 0.015,
                "open_drawer" : lambda s, p: on_table(s) and s[1] >= 0.15,
                "grip_blue_contrast" : lambda s, p: on_table(s) and s[1] >= 0.15,
                "grip_blue" : lambda s, p: blue(s, p, "held") and s[1] >= 0.15,
                "release_blue_in_drawer_contrast" : lambda s, p: on_table(s),
                "release_blue_in_drawer" : lambda s, p: blue(s, p, ["drawer", "released", "resting"]) and s[1] >= 0.15,
                "close_drawer" : lambda s, p: blue(s, p, "drawer") and s[1] < 0.015}
    for step in manifest["steps"]:
        path = os.path.join(folder, step["name"] + ".hdf5")
        with h5py.File(path, "r") as file:
            for demo in file["data"]:
                s, p = file[f"data/{demo}/obs/states"][-1], file[f"data/{demo}/obs/proprio"][-1]
                assert [c for c, i in BLOCK_INDEX.items() if s[i] < 2.0] == ["blue"], f"{path} {demo}: not exactly one block in view"
                assert expected[step["name"]](s, p), f"{path} {demo}: the blue block or the drawer isn't what the story says"
                for joint in [0, 3, 5]: # door, switch and green light as in the reset
                    assert s[joint] == pytest.approx(reset[joint]), f"{path} {demo}: joint {joint} differs from the reset"


def test_made_single_block_task_succeeds_in_order():
    config = load(os.path.join(made("single_block_drawer"), "experiment.json"))
    assert [step["name"] for step in config["goal_sequence"]] == ["open_drawer", "grip_blue", "release_blue_in_drawer", "close_drawer"]
    tracker = SequentialGoalTracker(config)
    state = fixed_reset(config["env_setup"])
    path = list()
    def at(drawer = None, blue = None, tcp = None, held = False):
        s = (path[-1][0] if path else state).copy()
        if drawer is not None:
            s[1] = drawer
        if blue is not None:
            s[12:15] = blue
        p = proprio(tcp if tcp is not None else [0.4, -0.3, 0.6], width = 0.04 if held else 0.08, gripper = -1.0 if held else 1.0)
        path.append((s, p))
    at(); at(drawer = 0.10); at(drawer = 0.19, tcp = [0.175, -0.39, 0.40]) # open
    at(blue = [0.30, -0.12, 0.46], tcp = [0.30, -0.12, 0.47], held = True) # gripped
    at(blue = [0.30, -0.12, 0.53], tcp = [0.30, -0.12, 0.53], held = True) # lifted
    at(blue = [0.18, -0.28, 0.3624], tcp = [0.18, -0.28, 0.45]) # in the drawer, let go
    at(drawer = 0.10, blue = [0.18, -0.19, 0.3624]); at(drawer = 0.0, blue = [0.18, -0.09, 0.3624], tcp = [0.175, -0.2, 0.40]) # closed
    done = [tracker.update(s, p) for s, p in path]
    assert tracker.success and done[-1] and tracker.steps_completed == 4


@pytest.mark.parametrize("name", CHAINS)
def test_before_examples_of_a_made_task(name):
    """Each before_example is its condition with nothing but the joint of the step changed, back to where it was before the step."""
    import h5py
    folder = made(name)
    joint = {"drawer" : 1, "door" : 0, "switch" : 3, "button" : 5}
    for step in load(os.path.join(folder, "experiment.json"))["goal_sequence"]:
        index, rising = joint[step["behaviors"][0].split("_")[0]], step["behaviors"][0].split("_")[1] in ["open", "left", "on"]
        with h5py.File(step["pos_examples"], "r") as goal, h5py.File(step["switch"]["before_examples"], "r") as before:
            assert sorted(goal["data"]) == sorted(before["data"])
            for demo in goal["data"]:
                a, b = [file[f"data/{demo}/obs/states"][-1] for file in [goal, before]]
                assert np.allclose(*[file[f"data/{demo}/obs/proprio"][-1] for file in [goal, before]])
                assert (a[index] > b[index]) == rising and abs(a[index] - b[index]) > 0.05
                may_differ = {index} | ({2} if index == 5 else set()) | ({4} if index == 3 else set()) # the button, the lightbulb
                assert set(np.flatnonzero(~np.isclose(a, b, atol = 1e-6))) <= may_differ
                assert (goal[f"data/{demo}/obs/third_person"][-1] != before[f"data/{demo}/obs/third_person"][-1]).any()
