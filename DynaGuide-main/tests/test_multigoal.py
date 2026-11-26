"""
Tests of multi-goal guidance: the env side (core/goal_sequence.py, block checks of core/guidance_conditions.py) and the DynaGuide side
(core/multigoal_guidance.py). Run from DynaGuide-main:

    python -m pytest tests/test_multigoal.py -v

The tests marked slow load the dynamics model (CPU is fine, about a minute). Skip them with -m "not slow".
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO) # the paths of the configs are relative to DynaGuide-main

from core.calvin_behaviors import GoalTracker, detect_behaviors
from core.goal_sequence import (SequentialGoalTracker, parse_goal_sequence, replay_sequence, summarize_sequence,
                                validate_multigoal_config)
from core.guidance_conditions import StateView, find_frames, read_recordings, state_matches
from core.multigoal_guidance import (CHANGE_DEFAULTS, GoalStepGuidance, MultiGoalGuidance, PATCH_DIM, PATCHES_DEFAULTS, SWITCH_DEFAULTS,
                                     SwitchCriterion, change_score, check_pairs, patches_score, switch_settings)

VALIDATION = "datasets/calvin_debug_dataset/validation"

# --- helpers --------------------------------------------------------------------------------------------------------------------------

def scene(drawer = 0.0, door = 0.0, switch = 0.0, light = 0.0, red = (0.05, -0.09, 0.46), blue = (0.175, -0.09, 0.46), pink = (0.30, -0.09, 0.46)):
    state = np.zeros(24)
    state[0], state[1], state[3], state[5] = door, drawer, switch, light
    state[6:9], state[12:15], state[18:21] = red, blue, pink
    return state


def robot(tcp = (0.0, 0.3, 0.6), width = 0.08, gripper_action = 1.0):
    proprio = np.zeros(15)
    proprio[:3], proprio[6], proprio[14] = tcp, width, gripper_action
    return proprio


DRAWER_BLOCK_DRAWER = {
    "env_setup" : {"drawer" : "closed"},
    "goal_sequence" : [
        {"name" : "open", "behaviors" : ["drawer_open"]},
        {"name" : "blue_in", "where" : {"blocks" : {"blue" : ["drawer", "released"]}}},
        {"name" : "close", "behaviors" : ["drawer_close"]},
    ]
}


def run(tracker, frames):
    for state, proprio in frames:
        if tracker.update(state, proprio):
            break
    return tracker

# --- block checks ------------------------------------------------------------------------------------------------------------------------

def test_block_words_and_specs():
    state = scene(drawer = 0.18, red = (0.05, -0.09, 0.46), blue = (0.05, -0.09, 0.50), pink = (0.2, -0.27, 0.3624))
    far = robot()
    assert state_matches(state, far, {"blocks" : {"red" : "table_left"}})
    assert not state_matches(state, far, {"blocks" : {"red" : "table_center"}})
    assert state_matches(state, far, {"blocks" : {"blue" : {"on" : "red"}}})
    assert not state_matches(state, far, {"blocks" : {"red" : {"on" : "blue"}}})
    assert state_matches(state, far, {"blocks" : {"pink" : ["drawer", "released", "resting"]}})
    assert state_matches(state, far, {"blocks" : {"pink" : {"near" : [0.2, -0.27, 0.36], "radius" : 0.02}}})
    assert state_matches(state, far, {"blocks" : {"red" : {"near" : "blue", "radius" : 0.05}}})
    holding = robot(tcp = (0.2, -0.27, 0.37), width = 0.04, gripper_action = -1.0)
    assert state_matches(state, holding, {"blocks" : {"pink" : "held"}})
    assert not state_matches(state, holding, {"blocks" : {"pink" : "released"}})


def test_block_spec_errors():
    view = StateView(scene(), robot())
    for bad in [{"blocks" : {"red" : "floating"}}, {"blocks" : {"red" : {"on" : "red"}}}, {"blocks" : {"red" : {"near" : [1, 2]}}},
                {"blocks" : {"green" : "table"}}, {"blocks" : {"red" : {"near" : "blue", "on" : "pink"}}}]:
        with pytest.raises(ValueError):
            find_frames(view, bad)


def test_lift_event():
    start = scene()
    up = scene(blue = (0.175, -0.09, 0.52))
    assert "blue_lift" in detect_behaviors(start, up, robot(tcp = (0.175, -0.09, 0.53)))
    assert "blue_lift" not in detect_behaviors(start, up, robot()) # nobody is at the block


def test_single_goal_tracker_unchanged():
    tracker = GoalTracker({"target_behaviors" : ["drawer_open"]})
    run(tracker, [(scene(drawer = d), robot()) for d in [0.0, 0.08, 0.16]])
    assert tracker.behaviors == ["drawer_open"]

# --- the sequential goal tracker ----------------------------------------------------------------------------------------------------------

def test_sequence_in_order():
    inside = (0.175, -0.27, 0.3624) # in the open drawer
    frames = [(scene(drawer = 0.0), robot()), (scene(drawer = 0.1), robot()), (scene(drawer = 0.18), robot()),   # open, done at t = 2
              (scene(drawer = 0.18, blue = (0.175, -0.2, 0.5)), robot(tcp = (0.175, -0.2, 0.5), width = 0.04, gripper_action = -1)), # carried
              (scene(drawer = 0.18, blue = inside), robot(tcp = inside, width = 0.04, gripper_action = -1)),  # in the drawer, still held
              (scene(drawer = 0.18, blue = inside), robot()),                                              # let go, done at t = 5
              (scene(drawer = 0.05, blue = inside), robot()), (scene(drawer = 0.0, blue = inside), robot())] # closed, done at t = 7
    tracker = run(SequentialGoalTracker(DRAWER_BLOCK_DRAWER), frames)
    summary = tracker.summary()
    assert summary["success"] and summary["steps_completed"] == 3
    assert [step["completed_at"] for step in summary["steps"]] == [2, 5, 7]
    assert tracker.behaviors[:2] == ["drawer_open", "blue_displace"] and "drawer_close" in tracker.behaviors


def test_open_and_close_both_count():
    """The drawer can't be open and closed at once, but opened in one step and closed in the next is a success."""
    config = {"env_setup" : {"drawer" : "closed"}, "goal_sequence" : [{"behaviors" : ["drawer_open"]}, {"behaviors" : ["drawer_close"]}]}
    tracker = run(SequentialGoalTracker(config), [(scene(drawer = d), robot()) for d in [0.0, 0.2, 0.2, 0.1, 0.0]])
    assert tracker.success and [s["completed_at"] for s in tracker.summary()["steps"]] == [1, 4]


def test_order_is_strict():
    """The second goal (switch on) already happened before the first (drawer open): it must happen again after."""
    config = {"env_setup" : {"drawer" : "closed", "switch" : "off"},
              "goal_sequence" : [{"behaviors" : ["drawer_open"]}, {"behaviors" : ["switch_on"]}]}
    frames = [(scene(), robot()), (scene(switch = 0.09), robot()), (scene(drawer = 0.2, switch = 0.09), robot())]
    tracker = run(SequentialGoalTracker(config), frames)
    assert tracker.steps_completed == 1 and not tracker.done
    run(tracker, [(scene(drawer = 0.2, switch = 0.0), robot()), (scene(drawer = 0.2, switch = 0.09), robot())])
    assert tracker.success


def test_several_goals_in_one_step():
    config = {"env_setup" : {"drawer" : "closed", "switch" : "off"},
              "goal_sequence" : [{"behaviors" : ["drawer_open", "switch_on"]}, {"behaviors" : ["drawer_close"]}]}
    tracker = run(SequentialGoalTracker(config), [(scene(), robot()), (scene(drawer = 0.2), robot())])
    assert tracker.steps_completed == 0 # only half of the step
    run(tracker, [(scene(drawer = 0.2, switch = 0.09), robot())])
    assert tracker.steps_completed == 1


def test_any_behaviors_and_blocks_moved():
    config = {"env_setup" : {}, "goal_sequence" : [{"any_behaviors" : ["door_left", "switch_on"]}, {"blocks_moved" : {"red" : 0.05}}]}
    tracker = run(SequentialGoalTracker(config), [(scene(), robot()), (scene(switch = 0.09), robot()),
                                                  (scene(switch = 0.09, red = (0.08, -0.09, 0.46)), robot()),
                                                  (scene(switch = 0.09, red = (0.12, -0.09, 0.46)), robot())])
    assert [s["completed_at"] for s in tracker.summary()["steps"]] == [1, 3]


def test_avoid_and_timeout():
    config = {"env_setup" : {"drawer" : "closed"}, "avoid_behaviors" : ["door_left"],
              "goal_sequence" : [{"behaviors" : ["drawer_open"]}, {"behaviors" : ["drawer_close"], "max_steps" : 2}]}
    tracker = run(SequentialGoalTracker(config), [(scene(), robot()), (scene(door = 0.27), robot())])
    assert tracker.done and not tracker.success and "door_left" in tracker.failure
    tracker = run(SequentialGoalTracker(config), [(scene(drawer = d), robot()) for d in [0.0, 0.2, 0.2, 0.2, 0.2]])
    assert tracker.steps_completed == 1 and "took more than" in tracker.failure


def test_validation_errors():
    with pytest.raises(ValueError, match = "drawer_close needs"):
        validate_multigoal_config({"env_setup" : {"drawer" : "closed"}, "goal_sequence" : [{"behaviors" : ["drawer_close"]}]})
    with pytest.raises(ValueError, match = "drawer_open needs"): # opened twice without closing in between
        validate_multigoal_config({"env_setup" : {}, "goal_sequence" : [{"behaviors" : ["drawer_open"]}, {"behaviors" : ["drawer_open"]}]})
    for bad in [{"env_setup" : {}, "goal_sequence" : []},
                {"env_setup" : {}, "goal_sequence" : [{"name" : "a"}]},                                   # no goal
                {"env_setup" : {}, "goal_sequence" : [{"behaviors" : ["drawer_opne"]}]},                  # typo
                {"env_setup" : {}, "goal_sequence" : [{"behaviors" : ["drawer_open"], "colour" : 1}]},    # unknown key
                {"env_setup" : {}, "target_behaviors" : ["drawer_open"], "goal_sequence" : [{"behaviors" : ["drawer_open"]}]}]:
        with pytest.raises(ValueError):
            validate_multigoal_config(bad)
    validate_multigoal_config(DRAWER_BLOCK_DRAWER)


def test_example_configs_are_valid():
    import glob, json
    from core.calvin_utils import condition_files
    files = sorted(glob.glob("calvin_exp_configs_examples/multigoal/*.json"))
    assert len(files) > 0
    for path in files:
        with open(path) as f:
            config = json.load(f)
        if "goal_sequence" not in config: # a plan
            continue
        validate_multigoal_config(config)
        for entry in config["goal_sequence"]:
            switch_settings(config, entry)
            condition_files(entry["pos_examples"])


def test_replay_recorded_sequence():
    """The debug validation recording opens the drawer, puts the pink and the blue block in, and closes the drawer."""
    recording = read_recordings([VALIDATION])[0]
    summary = replay_sequence(recording.states, recording.proprio, DRAWER_BLOCK_DRAWER)
    assert summary["success"], summary
    opened, placed, closed = [step["completed_at"] for step in summary["steps"]]
    assert 226 <= opened <= 307 and 307 < placed < 575 and 575 <= closed <= 661
    # the other way round can't succeed in this recording: the blocks only go in after the drawer opened
    # (until frame 1100, later the drawer opens again)
    reverse = {"env_setup" : {}, "goal_sequence" : [{"where" : {"blocks" : {"blue" : ["drawer", "released"]}}}, {"behaviors" : ["drawer_open"]}]}
    assert replay_sequence(recording.states[:1100], recording.proprio[:1100], reverse)["steps_completed"] == 1


def test_summary_and_switch_statistics():
    episodes = [{"success" : True, "steps_completed" : 2, "failure" : None, "behaviors" : ["drawer_open", "drawer_close"],
                 "steps" : [{"completed_at" : 10}, {"completed_at" : 30}]},
                {"success" : False, "steps_completed" : 1, "failure" : "x", "behaviors" : ["drawer_open"],
                 "steps" : [{"completed_at" : 12}, {"completed_at" : None}]}]
    logs = [{"switches" : [{"t" : 14, "from_step" : 0}]}, {"switches" : [{"t" : 5, "from_step" : 0}]}]
    config = {"env_setup" : {}, "goal_sequence" : [{"name" : "open", "behaviors" : ["drawer_open"]}, {"name" : "close", "behaviors" : ["drawer_close"]}]}
    summary = summarize_sequence(episodes, config, logs)
    assert summary["Success_Rate"] == 0.5 and summary["Mean_Steps_Completed"] == 1.5
    assert summary["Step_Reach_Rate"] == {"open" : 1.0, "close" : 0.5}
    assert summary["Switching"]["open"] == {"switched" : 2, "too_early" : 1, "missed" : 0, "mean_delay" : 4.0, "max_delay" : 4}

# --- the DynaGuide side ------------------------------------------------------------------------------------------------------------------

class FakeModel:
    """Embeds an image as its mean value, so the tests control what the switch sees."""
    def state_embedding(self, state):
        return state["third_person"].mean(dim = (1, 2, 3)).reshape(-1, 1, 1)

    def state_action_embedding(self, state, actions):
        return self.state_embedding(state)


def fake_obs(value):
    return {"third_person" : np.full((2, 3, 8, 8), value / 255.0), "proprio" : np.zeros((2, 15))}


def fake_guidance(steps_switch, goal_values, contrast_values):
    steps = list()
    for number, (switch, goal, contrast) in enumerate(zip(steps_switch, goal_values, contrast_values)):
        as_tensor = lambda values: None if values is None else torch.tensor(values, dtype = torch.float32).reshape(-1, 1)
        steps.append(GoalStepGuidance(f"step{number}", lambda s, a, n = number: torch.full((1,), float(n)), None, None, alpha = 1.0,
                                      switch = switch, goal_embeddings = as_tensor(goal), contrast_embeddings = as_tensor(contrast)))
    return MultiGoalGuidance(FakeModel(), steps, "third_person", "cpu")


def test_peak_switch():
    """Done when the contrast score c = d_contrast - d_goal rose above the margin and stopped rising for patience checks."""
    switch = {**SWITCH_DEFAULTS, "patience" : 2, "min_steps" : 0, "margin" : 30, "tolerance" : 5}
    guidance = fake_guidance([switch, switch], [[100.0], [200.0]], [[0.0], [100.0]]) # c = 2 * value - 100 for value in [0, 100]
    assert guidance.steps[0].criterion.criterion == "peak"
    assert float(guidance(None, None)) == 0
    observations = [20, 40, 60, 70, 71, 72] # c: -60, -20 (crossed), 20, 40, then 42 and 44 don't beat 40 by more than 5
    assert [guidance.observe(fake_obs(value), t) for t, value in enumerate(observations, 1)] == [False] * 5 + [True]
    assert guidance.current == 1 and float(guidance(None, None)) == 1
    assert guidance.log()["switches"][0]["t"] == 6
    assert not guidance.observe(fake_obs(250), 7)  # the last step never switches, the env ends the episode


def test_peak_needs_a_crossing():
    """A step that looks done from the start isn't skipped: c must have been <= 0 during the step."""
    switch = {**SWITCH_DEFAULTS, "patience" : 2, "min_steps" : 0}
    guidance = fake_guidance([switch, switch], [[100.0], [200.0]], [[0.0], [100.0]])
    assert not any(guidance.observe(fake_obs(90), t) for t in range(1, 10)) # c = 80 all the time
    assert not guidance.observe(fake_obs(10), 10)                            # c = -80, crossed
    assert [guidance.observe(fake_obs(90), t) for t in range(11, 14)] == [False, False, True]


def test_contrast_switch_with_patience():
    switch = {**SWITCH_DEFAULTS, "criterion" : "contrast", "patience" : 2, "min_steps" : 0, "margin" : 10}
    guidance = fake_guidance([switch, switch], [[100.0], [200.0]], [[0.0], [100.0]])
    assert not guidance.observe(fake_obs(20), 1)   # c = -60
    assert not guidance.observe(fake_obs(80), 2)   # c = 60, once
    assert not guidance.observe(fake_obs(50), 3)   # c = 0, the streak restarts
    assert not guidance.observe(fake_obs(80), 4)
    assert guidance.observe(fake_obs(90), 5)       # twice in a row


def test_min_steps_and_relative_fallback():
    switch = {**SWITCH_DEFAULTS, "patience" : 1, "min_steps" : 5, "ratio" : 1.0}
    guidance = fake_guidance([switch, switch], [[100.0, 110.0], None], [None, None]) # no contrast: relative, spread 10
    assert guidance.steps[0].criterion.criterion == "relative"
    assert not guidance.observe(fake_obs(100), 3)  # done, but before min_steps
    assert not guidance.observe(fake_obs(80), 6)   # 20 away, more than the spread
    assert guidance.observe(fake_obs(105), 7)


def test_oracle_switch():
    oracle = {**SWITCH_DEFAULTS, "mode" : "oracle", "delay" : 0}
    guidance = fake_guidance([oracle, oracle, oracle], [None] * 3, [None] * 3)
    assert not guidance.observe(fake_obs(0), 1, oracle_steps_completed = 0)
    assert guidance.observe(fake_obs(0), 2, oracle_steps_completed = 1)
    assert guidance.observe(fake_obs(0), 3, oracle_steps_completed = 2) and guidance.current == 2
    guidance.reset()
    assert guidance.current == 0 and guidance.log()["switches"] == []


def test_oracle_waits_delay():
    oracle = {**SWITCH_DEFAULTS, "mode" : "oracle", "delay" : 3}
    guidance = fake_guidance([oracle, oracle], [None] * 2, [None] * 2)
    assert not guidance.observe(fake_obs(0), 5, oracle_steps_completed = 1) # the env sees it done here
    assert not guidance.observe(fake_obs(0), 7, oracle_steps_completed = 1)
    assert guidance.observe(fake_obs(0), 8, oracle_steps_completed = 1)
    assert guidance.log()["switches"][0]["env_done"] == 5
    assert SWITCH_DEFAULTS["delay"] == 10


def test_done_looser_than_completed():
    from core.calvin_behaviors import COMPLETED, DONE, detect_behaviors
    for name in COMPLETED:
        rising = name in ["door_left", "drawer_open", "switch_on"]
        assert (DONE[name] <= COMPLETED[name]) if rising else (DONE[name] >= COMPLETED[name])
    start, state = np.zeros(24), np.zeros(24)
    start[3], state[3] = 0.088, 0.010 # the switch 89% of the way off: done now, not before
    assert "switch_off" in detect_behaviors(start, state, np.zeros(15))


def test_switch_settings():
    config = {"switch" : {"patience" : 5}}
    assert switch_settings(config, {})["patience"] == 5
    assert switch_settings(config, {"switch" : {"patience" : 1, "mode" : "oracle"}})["mode"] == "oracle"
    for bad in [{"mode" : "magic"}, {"criterion" : "threshold"}, {"patiense" : 2}]:
        with pytest.raises(ValueError):
            switch_settings({"switch" : bad}, {})

# --- the change criterion ---------------------------------------------------------------------------------------------------------------

CHANGE = {**SWITCH_DEFAULTS, **CHANGE_DEFAULTS}


def test_change_score():
    """-1 as before, +1 as in the goal, and what isn't part of the change adds nothing."""
    torch.manual_seed(0)
    robot = torch.randn(4, 50) * 10 # the robot pose of each pair, the same before and after
    goal, before = torch.cat([robot, torch.full((4, 3), 5.0)], dim = 1), torch.cat([robot, torch.zeros(4, 3)], dim = 1)
    score = lambda query: change_score(goal, before, query).numpy()
    assert np.allclose(score(goal), 1, atol = 1e-4) and np.allclose(score(before), -1, atol = 1e-4)
    elsewhere = lambda value: torch.cat([torch.randn(5, 50) * 100, torch.full((5, 3), value)], dim = 1) # the robot anywhere else
    assert np.allclose(score(elsewhere(5.0)), 1, atol = 1e-3) and np.allclose(score(elsewhere(0.0)), -1, atol = 1e-3)
    assert np.allclose(score(elsewhere(2.5)), 0, atol = 1e-3) # halfway
    hidden = torch.cat([robot[:1], torch.zeros(1, 3)], dim = 1) # a pair in which the arm hides the object: no change, no say
    assert np.allclose(change_score(torch.cat([goal, hidden]), torch.cat([before, hidden]), goal).numpy(), 1, atol = 1e-4)


def switch_at(scores, **settings):
    criterion = SwitchCriterion({**CHANGE, **settings}, "change")
    return next((t for t, score in enumerate(scores) if criterion.update_score(score)), None)


def test_change_waits_until_the_score_stopped_rising():
    rise = list(np.linspace(-0.8, 0.7, 31))                        # 0.05 per check, above the margin from check 21
    assert switch_at(rise) is None                                 # still moving
    assert switch_at(rise + [0.7] * 20, patience = 3) in [32, 33]  # comes to rest
    assert switch_at([0.1] * 50) is None                           # rests below the margin
    assert switch_at([-0.5] * 50) is None
    assert switch_at([0.6] * 50) == 10                             # the goal holds already, nothing to wait for but patience
    assert switch_at([0.6, 0.6, -0.4, -0.4] + rise) is None        # undone and done again: the first time doesn't count
    assert switch_at([0.6, 0.6, -0.4, 0.5, 0.5, 0.5], patience = 2) == 5


def test_change_settings():
    before = {"before_examples" : "x.hdf5"}
    settings = switch_settings({}, {"switch" : before}) # patches is the default since 2026-11-26
    assert [settings[key] for key in ["criterion", "margin", "tolerance", "patience"]] == ["patches", 0.6, 0.05, 10]
    settings = switch_settings({"switch" : {"criterion" : "change"}}, {"switch" : before})
    assert [settings[key] for key in ["criterion", "margin", "tolerance", "patience"]] == ["change", 0.2, 0.05, 10]
    assert switch_settings({}, {})["criterion"] == "peak" and switch_settings({}, {})["margin"] == 30
    assert switch_settings({"switch" : {"margin" : 0.3}}, {"switch" : before})["margin"] == 0.3
    previous = switch_settings({"switch" : {"criterion" : "peak"}}, {"switch" : before}) # the older switch, to compare
    assert previous["criterion"] == "peak" and previous["margin"] == 30 and previous["patience"] == 5
    fallback = switch_settings({"switch" : {"criterion" : "change", "margin" : 0.3}}, {}) # a block step among joint steps
    assert fallback["criterion"] == "peak" and fallback["margin"] == 30
    assert switch_settings({"switch" : {"embedding" : "predicted"}}, {"switch" : before})["criterion"] == "peak"
    for bad in [{"criterion" : "change"}, {**before, "margin" : 30}, {**before, "criterion" : "change", "embedding" : "predicted"}]:
        with pytest.raises(ValueError):
            switch_settings({}, {"switch" : bad})


def test_change_switch_moves_on():
    """In the whole guidance, and the step without before_examples falls back to the older switch."""
    switch = {**CHANGE, "patience" : 2, "min_steps" : 0}
    values = lambda *numbers: torch.tensor(numbers, dtype = torch.float32).reshape(-1, 1)
    steps = [GoalStepGuidance("open", None, None, None, 1.0, switch, values(100, 102), None, values(0, 2)),
             GoalStepGuidance("last", None, None, None, 1.0, switch, None, None)]
    guidance = MultiGoalGuidance(FakeModel(), steps, "third_person", "cpu")
    done = [guidance.observe(fake_obs(value), t) for t, value in enumerate([0, 20, 60, 90, 100, 101, 101], 1)]
    assert done == [False] * 6 + [True] and guidance.log()["switches"][0]["reason"] == "change"
    assert guidance.log()["trace"][0][2] == pytest.approx(-1.02, abs = 0.01)
    with pytest.raises(ValueError, match = "its own"):
        GoalStepGuidance("open", None, None, None, 1.0, switch, values(100, 102), None, values(0))


def test_patches_score():
    """Only the patches that change count, and the best fitting pair decides."""
    torch.manual_seed(0)
    def images(places, value, arm): # one image per place: 8 patches, the block at patch place, the arm everywhere else in patch 0
        x = torch.zeros(len(places), 8, PATCH_DIM)
        x[:, 0] = arm
        for image, place in zip(x, places):
            image[place] = value
        return x.flatten(start_dim = 1)
    places = [2, 3, 5] # three goal conditions, the block at three places
    goal, before = images(places, 4.0, 1.0), images(places, 0.0, 1.0)
    score = lambda query: patches_score(goal, before, query).numpy()
    assert np.allclose(score(goal), 1, atol = 1e-4) # one fitting place is enough, the other two places don't pull the score down
    assert np.allclose(score(images([7], 4.0, 1.0)), -1, atol = 1e-4) # a block where no condition has one
    assert np.allclose(score(images([2], 4.0, 50.0)), 1, atol = 1e-4) # the arm elsewhere
    assert np.allclose(score(images([2], 2.0, 1.0)), 0, atol = 1e-4)  # halfway
    assert change_score(goal, before, goal)[0] < 0 # the average of the pairs can't tell that the block is at one of the places
    same = before.clone()
    assert np.allclose(patches_score(same, same, before).numpy(), -1) # no pair shows a change
    weak = goal.clone().reshape(3, 8, PATCH_DIM)
    weak[:, 1] = 0.5 # a patch that changes little, 1/8 of the strongest one
    query = weak.clone()
    query[:, 1] = 30.0
    assert np.allclose(patches_score(weak.flatten(1), before, query.flatten(1)).numpy(), 1, atol = 1e-4) # doesn't count
    assert (patches_score(weak.flatten(1), before, query.flatten(1), changed = 0.05).numpy() != 1).any()


def test_patches_settings():
    before = {"before_examples" : "x.hdf5"}
    settings = switch_settings({"switch" : {"criterion" : "patches"}}, {"switch" : before})
    assert [settings[key] for key in ["criterion", "margin", "tolerance", "patience"]] == ["patches", 0.6, 0.05, 10]
    release = {**before, "default_criterion" : "patches"} # what make_task.py writes for a release step
    assert switch_settings({}, {"switch" : release})["criterion"] == "patches"
    assert switch_settings({"switch" : {"criterion" : "peak"}}, {"switch" : release})["criterion"] == "peak" # a plan's variant decides
    assert switch_settings({"switch" : {"criterion" : "patches", "margin" : 0.4}}, {})["margin"] == 30 # a step without pictures falls back
    for bad in [{**before, "criterion" : "patches", "margin" : 30}, {**before, "criterion" : "patches", "changed" : 2}]:
        with pytest.raises(ValueError):
            switch_settings({}, {"switch" : bad})


def test_patches_switch_moves_on():
    switch = {**SWITCH_DEFAULTS, **PATCHES_DEFAULTS, "patience" : 2, "min_steps" : 0}
    image = lambda value: torch.full((1, PATCH_DIM), float(value))
    steps = [GoalStepGuidance("open", None, None, None, 1.0, switch, image(100), None, image(0)),
             GoalStepGuidance("last", None, None, None, 1.0, switch, None, None)]
    assert steps[0].criterion.criterion == "patches"
    done = [steps[0].check(image(value))[0] for value in [0, 40, 90, 100, 101, 101]]
    assert done == [False] * 5 + [True]


def test_before_examples_must_be_twins():
    class Conditions:
        def __init__(self, proprio):
            self.lengths_list, self.proprio = [1] * len(proprio), proprio
        def get_labeled_item(self, index):
            return {"proprio" : np.full(15, self.proprio[index])}, None, None
    check_pairs("open", Conditions([1.0, 2.0]), Conditions([1.0, 2.0]))
    for wrong in [[1.0], [2.0, 1.0]]:
        with pytest.raises(ValueError):
            check_pairs("open", Conditions([1.0, 2.0]), Conditions(wrong))

# --- with the dynamics model -------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope = "module")
def dynamics_model():
    from core.dynamics_models import FinalStatePredictionDino
    model = FinalStatePredictionDino(7, 16, cameras = ["third_person"], reconstruction = True, proprio = "proprio", proprio_dim = 15)
    model.load_state_dict(torch.load("dynamics_model_5900.pth", map_location = "cpu"))
    return model.eval()


@pytest.mark.slow
def test_patch_cache_gives_the_same_gradient(dynamics_model):
    """calculate_classifier_guidance encodes the image once per observation, this must not change the gradient."""
    from core.dynaguide import calculate_classifier_guidance
    from core.embedder_datasets import load_conditions
    conditions = load_conditions("datasets/guidance_conditions/debug_all/drawer_open.hdf5", action_chunk_length = 16,
                                 cameras = ["third_person"], padding = True, pad_mode = "repeat", proprio = "proprio")
    guidance, good, _ = calculate_classifier_guidance(dynamics_model, conditions, scale = 1.0, main_camera = "third_person",
                                                      alpha = 30, max_examples = 4, device = "cpu")
    torch.manual_seed(0)
    states = {"third_person" : torch.rand(1, 2, 3, 200, 200), "proprio" : torch.rand(1, 2, 15)}
    first = guidance(states, torch.randn(1, 16, 7, requires_grad = True))
    actions = torch.randn(1, 16, 7, requires_grad = True)
    cached = guidance(states, actions) # the patches come from the cache now

    relevant = {"third_person" : states["third_person"][:, -1] * 255, "proprio" : states["proprio"][:, -1]}
    predicted = dynamics_model.state_action_embedding(relevant, actions).flatten(start_dim = 1)
    score = torch.logsumexp(-torch.cdist(good, predicted) / 30, dim = 0)
    direct = torch.autograd.grad(score, actions)[0]
    assert torch.allclose(cached, direct, atol = 1e-5) and first.shape == cached.shape

    other = {"third_person" : torch.rand(1, 2, 3, 200, 200), "proprio" : states["proprio"]} # a new observation is encoded again
    assert not torch.allclose(guidance(other, actions), cached)


@pytest.mark.slow
def test_multigoal_guidance_builds_from_conditions(dynamics_model):
    from core.embedder_datasets import load_conditions
    def conditions(path):
        return load_conditions(path, action_chunk_length = 16, cameras = ["third_person"], padding = True, pad_mode = "repeat", proprio = "proprio")
    switch = switch_settings({}, {})
    inputs = [{"name" : "open", "good_dataset" : conditions("datasets/guidance_conditions/debug_all/drawer_open.hdf5"), "bad_dataset" : None,
               "scale" : 1.0, "alpha" : 30, "switch" : switch},
              {"name" : "close", "good_dataset" : conditions("datasets/guidance_conditions/debug_all/drawer_close.hdf5"), "bad_dataset" : None,
               "scale" : 1.0, "alpha" : 30, "switch" : switch}]
    guidance = MultiGoalGuidance.build(dynamics_model, inputs, "third_person", "cpu", max_examples = 3)
    assert guidance.steps[0].goal_embeddings.shape == (3, 256 * 384)
    assert guidance.steps[0].contrast_embeddings is None and guidance.steps[0].criterion.criterion == "relative"
    assert guidance.steps[1].goal_embeddings is None # the last step needs no switch
    recording = read_recordings([VALIDATION])[0]
    image = recording.load_images("third_person", [0])[0]
    obs = {"third_person" : np.transpose(image, (2, 0, 1))[None] / 255.0, "proprio" : recording.proprio[:1]}
    guidance.observe(obs, 20)
    assert len(guidance.trace) == 1 and guidance.trace[0][3] > 0

    # with contrast conditions the default peak criterion is used, and at the start (drawer closed) it looks like the contrast
    inputs[0]["contrast_dataset"] = inputs[1]["good_dataset"]
    guidance = MultiGoalGuidance.build(dynamics_model, inputs, "third_person", "cpu", max_examples = 3)
    assert guidance.steps[0].criterion.criterion == "peak"
    guidance.observe(obs, 20)
    t, step, score, d_goal, d_contrast = guidance.trace[0]
    assert score == d_contrast - d_goal and score < 0


# --- the drawer check and on_avoid -------------------------------------------------------------------------------------------------------

def test_block_outside_the_drawer_is_not_in_it():
    far = robot()
    inside = scene(drawer = 0.2, blue = (0.18, -0.285, 0.3624)) # y with the drawer pushed in is -0.085
    assert state_matches(inside, far, {"blocks" : {"blue" : ["drawer", "released", "resting"]}})
    for position in [(0.43, -0.317, 0.35),   # next to the drawer, low. A rollout of blue_drawer_end counted this as a success
                     (0.18, -0.50, 0.35),    # in front of it
                     (0.18, -0.085, 0.35)]:  # behind it, the drawer is 20 cm further out
        assert not state_matches(scene(drawer = 0.2, blue = position), far, {"blocks" : {"blue" : "drawer"}})
        assert not state_matches(scene(drawer = 0.2, blue = position), far, {"blocks" : {"blue" : "resting"}})


PRESS_FIRST = [(scene(), robot()), (scene(light = 1.0), robot()), (scene(light = 1.0, drawer = 0.2), robot())]


def test_on_avoid_end_or_record():
    config = {"env_setup" : {"drawer" : "closed"}, "goal_sequence" : [{"name" : "open", "behaviors" : ["drawer_open"], "avoid_behaviors" : ["button_on"]}]}
    ended = run(SequentialGoalTracker(config), PRESS_FIRST)
    assert ended.failure == "button_on during step open" and not ended.success and ended.t == 1 # the default, as before
    recorded = run(SequentialGoalTracker({**config, "on_avoid" : "record"}), PRESS_FIRST)
    assert recorded.success and recorded.failure is None and recorded.violations == [(1, "button_on", "open")]
    summary = summarize_sequence([ended.summary(), recorded.summary()], config)
    assert summary["Num_Success"] == 1 and summary["Num_Success_Without_Violation"] == 0
    assert summary["Violations"] == {"button_on during step open" : 2}
    step = {**config["goal_sequence"][0], "on_avoid" : "end"} # a step overrides the config
    assert run(SequentialGoalTracker({**config, "on_avoid" : "record", "goal_sequence" : [step]}), PRESS_FIRST).failure is not None


def test_avoid_counts_from_the_start_of_the_step():
    """The press that completes the button step isn't a violation of the next step, a later one is."""
    config = {"env_setup" : {"green_light" : "off"}, "on_avoid" : "record", "goal_sequence" : [
        {"name" : "press", "behaviors" : ["button_on"]}, {"name" : "open", "behaviors" : ["drawer_open"], "avoid_behaviors" : ["button_on", "button_off"]}]}
    frames = [(scene(), robot()), (scene(light = 1.0), robot()), (scene(light = 1.0), robot())]
    assert run(SequentialGoalTracker(config), frames).violations == []
    assert run(SequentialGoalTracker(config), frames + [(scene(light = 0.0), robot())]).violations == [(3, "button_off", "open")]


def test_on_avoid_errors():
    step = {"behaviors" : ["drawer_open"]}
    with pytest.raises(ValueError, match = "on_avoid"):
        validate_multigoal_config({"env_setup" : {}, "on_avoid" : "stop", "goal_sequence" : [step]})
    with pytest.raises(ValueError, match = "to be reached and to be avoided"):
        validate_multigoal_config({"env_setup" : {}, "avoid_behaviors" : ["drawer_open"], "goal_sequence" : [step]})


def test_on_avoid_of_a_single_goal():
    config = {"target_behaviors" : ["drawer_open"], "avoid_behaviors" : ["button_on"], "env_setup" : {}}
    tracker = run(GoalTracker(config), PRESS_FIRST)
    assert tracker.behaviors == ["button_on"] # ended there
    tracker = run(GoalTracker({**config, "on_avoid" : "record"}), PRESS_FIRST)
    assert tracker.behaviors == ["button_on", "drawer_open"]
