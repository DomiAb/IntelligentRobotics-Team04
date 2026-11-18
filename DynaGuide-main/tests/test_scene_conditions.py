"""
Tests of the block poses and the conditions rendered into one scene (dynaguide_scripts/extract_block_poses.py,
dynaguide_scripts/render_scene_conditions.py) and of the single-block task config. Run from DynaGuide-main:

    python -m pytest tests/test_scene_conditions.py -v

Needs the calvin_env package (PYTHONPATH with calvin/calvin_env), the scripts import the simulator. No simulator is started.
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
from extract_block_poses import candidate_frames, frame_masks
from render_scene_conditions import compose, held_joint_moved, scene_state

CONFIG = "calvin_exp_configs_examples/multigoal/single_block_drawer.json"
SPEC = "calvin_exp_configs_examples/scene_conditions/single_block_drawer.json"


def load(path):
    with open(path) as f:
        return json.load(f)


def proprio(tcp, width = 0.08, gripper = 1.0):
    p = np.zeros(15)
    p[0:3], p[6], p[14] = tcp, width, gripper
    return p


def test_scene_is_the_reset_of_the_experiment():
    spec, config = load(SPEC), load(CONFIG)
    assert spec["scene"] == config["env_setup"]
    state = scene_state(spec["scene"])
    assert np.allclose(state[BLOCK_INDEX["blue"] : BLOCK_INDEX["blue"] + 3], np.array(PLACES["table_right"].low) + [0.0, -0.03, 0.0])
    assert np.allclose(state[BLOCK_INDEX["red"] : BLOCK_INDEX["red"] + 3], HIDDEN_POS["red"])
    assert state[0] == pytest.approx(0.27) and state[1] == 0.0 and state[3] == 0.0 and state[5] == 0.0


def test_scene_must_be_fixed():
    scene = dict(load(SPEC)["scene"])
    scene["blocks"] = {**scene["blocks"], "blue": "table"} # a region, every reset would differ
    with pytest.raises(ValueError):
        scene_state(scene)
    with pytest.raises(ValueError):
        scene_state({k: v for k, v in load(SPEC)["scene"].items() if k != "switch"})


def test_frame_masks_lifted_and_released():
    states = np.zeros((3, 24))
    states[:, 1] = 0.18 # drawer open
    for name, index in BLOCK_INDEX.items():
        states[:, index : index + 3] = HIDDEN_POS[name]
    states[0, 12:15] = [0.17, -0.09, 0.52] # held up above the table
    states[1, 12:15] = [0.17, -0.25, 0.3624] # on the drawer floor
    states[2, 12:15] = [0.17, -0.09, 0.46] # on the table
    prop = np.stack([proprio([0.17, -0.09, 0.53], width = 0.04, gripper = -1.0), proprio([0.17, -0.25, 0.45]),
                     proprio([0.17, -0.09, 0.47], width = 0.04, gripper = -1.0)])
    masks = frame_masks(states, prop)
    assert list(masks[("block_lifted", "blue")]) == [True, False, False]
    assert list(masks[("block_in_drawer_released", "blue")]) == [False, True, False]
    states[1, 1] = 0.12 # the drawer not open as far as COMPLETED
    assert not frame_masks(states, prop)[("block_in_drawer_released", "blue")][1]


def test_candidate_frames():
    mask = np.zeros(30, dtype = bool)
    mask[2:20] = True
    mask[25] = True
    assert candidate_frames(mask, 8) == [3, 11, 19, 25]


def test_compose_moves_the_block_of_the_pose_and_follows_the_drawer():
    scene = scene_state(load(SPEC)["scene"])
    pose_states = np.zeros(24)
    pose_states[1] = 0.20
    pose_states[18:24] = [0.2, -0.27, 0.3624, 0, 0, 0.5] # the pink block in the drawer
    pose = {"states" : pose_states, "proprio" : proprio([0.2, -0.27, 0.45]), "block" : "pink", "file" : "x"}
    state, reason = compose(scene, "blue", pose, {"joints" : {"drawer" : 0.18}, "block" : "pose"})
    assert reason is None
    assert np.allclose(state[12:18], [0.2, -0.25, 0.3624, 0, 0, 0.5]) # moved in with the drawer, 0.02 less open
    assert np.allclose(state[18:21], HIDDEN_POS["pink"]) # the pink block stays out of the scene
    state, _ = compose(scene, "blue", pose, {"joints" : {"drawer" : "pose"}, "block" : "drawer_center"})
    assert np.allclose(state[12:15], [0.18, -0.085 - 0.20, PLACES["drawer_center"].low[2]])


def test_held_joint_is_not_moved():
    pose_states = np.zeros(24)
    pose_states[1] = 0.0 # closed drawer, gripper at its handle
    pose = {"states" : pose_states, "proprio" : proprio([0.175, -0.205, 0.40])}
    moved = pose_states.copy()
    moved[1] = 0.18
    assert held_joint_moved(pose, moved) == "drawer"
    assert held_joint_moved(pose, pose_states) is None


def test_single_block_task_succeeds_in_order():
    config = load(CONFIG)
    assert [step.name for step in validate_multigoal_config(config)] == ["open_drawer", "pick_block", "place_block_in_drawer", "close_drawer"]
    tracker = SequentialGoalTracker(config)
    state = scene_state(config["env_setup"])
    hand = proprio([0.4, -0.3, 0.6])
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
    at(blue = [0.175, -0.09, 0.47], tcp = [0.175, -0.09, 0.47], held = True)
    at(blue = [0.175, -0.09, 0.53], tcp = [0.175, -0.09, 0.53], held = True) # lifted
    at(blue = [0.18, -0.28, 0.3624], tcp = [0.18, -0.28, 0.45]) # in the drawer, let go
    at(drawer = 0.10, blue = [0.18, -0.19, 0.3624]); at(drawer = 0.0, blue = [0.18, -0.09, 0.3624], tcp = [0.175, -0.2, 0.40]) # closed
    done = [tracker.update(s, p) for s, p in path]
    assert tracker.success and done[-1] and tracker.steps_completed == 4


RENDERED = "datasets/guidance_conditions/single_block_drawer"


@pytest.mark.skipif(not os.path.isdir(RENDERED), reason = "render the conditions first, see run_single_block_drawer.bat")
def test_rendered_conditions_show_one_block_where_the_step_says():
    import h5py
    from core.guidance_conditions import StateView, find_frames
    config = load(CONFIG)
    start = scene_state(config["env_setup"])[12:15]
    where = {"drawer_closed_block_on_table" : lambda s, p: np.allclose(s[12:15], start, atol = 1e-3) and s[1] < 0.015,
             "drawer_open_block_on_table" : lambda s, p: np.allclose(s[12:15], start, atol = 1e-3) and s[1] >= 0.15,
             "block_lifted_drawer_open" : lambda s, p: s[1] >= 0.15 and find_frames(StateView(s, p), {"blocks" : {"blue" : "lifted"}})[0],
             "block_in_open_drawer_released" : lambda s, p: s[1] >= 0.15
                 and find_frames(StateView(s, p), {"blocks" : {"blue" : ["drawer", "released", "resting"]}})[0],
             "drawer_closed_block_in_drawer" : lambda s, p: s[1] < 0.015 and find_frames(StateView(s, p), {"blocks" : {"blue" : "drawer"}})[0]}
    for name, check in where.items():
        with h5py.File(os.path.join(RENDERED, name + ".hdf5"), "r") as file:
            for demo in file["data"]:
                s, p = file[f"data/{demo}/obs/states"][-1], file[f"data/{demo}/obs/proprio"][-1]
                assert [c for c, i in BLOCK_INDEX.items() if s[i] < 2.0] == ["blue"], f"{name} {demo}: not exactly one block in view"
                assert check(s, p), f"{name} {demo}: the blue block isn't where the step says"
                assert s[0] > 0.26 and s[3] < 0.01 and s[5] == 0.0, f"{name} {demo}: door, switch or light differ from the reset"
