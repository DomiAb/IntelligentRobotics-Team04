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
from core.multigoal_guidance import GoalStepGuidance, MultiGoalGuidance, SWITCH_DEFAULTS, switch_settings

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
    oracle = {**SWITCH_DEFAULTS, "mode" : "oracle"}
    guidance = fake_guidance([oracle, oracle, oracle], [None] * 3, [None] * 3)
    assert not guidance.observe(fake_obs(0), 1, oracle_steps_completed = 0)
    assert guidance.observe(fake_obs(0), 2, oracle_steps_completed = 1)
    assert guidance.observe(fake_obs(0), 3, oracle_steps_completed = 2) and guidance.current == 2
    guidance.reset()
    assert guidance.current == 0 and guidance.log()["switches"] == []


def test_switch_settings():
    config = {"switch" : {"patience" : 5}}
    assert switch_settings(config, {})["patience"] == 5
    assert switch_settings(config, {"switch" : {"patience" : 1, "mode" : "oracle"}})["mode"] == "oracle"
    for bad in [{"mode" : "magic"}, {"criterion" : "threshold"}, {"patiense" : 2}]:
        with pytest.raises(ValueError):
            switch_settings({"switch" : bad}, {})

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
