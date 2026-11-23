import h5py
import numpy as np
from collections import Counter

from core.calvin_utils import articulated_value, check_state_difference, validate_env_setup

"""
Goal conditions for CALVIN rollouts. The environment has no success signal of its own, so we detect what the robot achieved
from the states and compare this to the target_behaviors / avoid_behaviors of the experiment config. run_dynaguide.py only
asks this file whether the goal is decided, it doesn't look at the state itself.

A joint behavior counts when the joint starts on one side (ARTICULATED, the authors' segmentation thresholds of
data_processing_calvin/calvin_to_labeled_hdf5.py) and reaches the COMPLETED value close to its resting position on the other side.
The authors' end thresholds only show that the motion got going, e.g. a drawer that is half open. ARTICULATED stays as it is,
because the guidance conditions are cut and named with it (core/guidance_conditions.py, recombine_conditions.py).

The sanity_ behaviors replicate the original paper instead, to compare against the authors' numbers: the episode ends at the first
small change (check_state_difference) and is scored by what analyze_calvin_touch.py labels as the first interaction.
"""

BEHAVIORS = ["button_on", "button_off", "switch_on", "switch_off", "drawer_open", "drawer_close", "door_left", "door_right",
             "red_displace", "blue_displace", "pink_displace", "no_behavior"]

# state index, low, high, name when going from below low to above high, name when going from above high to below low
# the index is the one in the lowdim state, see generate_reset_state in calvin_utils.py
ARTICULATED = [(0, 0.05, 0.25, "door_left", "door_right"),
               (1, 0.05, 0.10, "drawer_open", "drawer_close"),
               (3, 0.01, 0.07, "switch_on", "switch_off"),
               (5, 0.2, 0.8, "button_on", "button_off")]

# where a behavior counts as finished, used by GoalTracker instead of the end threshold of ARTICULATED. Chosen from the resting
# positions in the debug set (in brackets), with a margin so a joint that stops a little short still counts:
# door right [0 to 0.025] and left [0.272 to 0.28], drawer closed [0] and open [0.178 to 0.217], switch off [0] and on [0.088]
# The button isn't here: green_light is 0 or 1, so ARTICULATED already means the light switched
COMPLETED = {"door_left" : 0.26, "door_right" : 0.035,
             "drawer_open" : 0.15, "drawer_close" : 0.015,
             "switch_on" : 0.08, "switch_off" : 0.008}

BLOCKS = {"red_displace" : 6, "blue_displace" : 12, "pink_displace" : 18} # index of the block position
BLOCK_MOVED = 0.01
BLOCK_ROBOT_NEAR = 0.1 # so we don't count a block that moves with the drawer

SANITY_BEHAVIORS = ["sanity_" + name for name in BEHAVIORS]
SANITY_ANY = "sanity_any" # any interaction counts, for the control runs of the paper


def get_behavior_goals(config):
    """Reads the goals of an experiment config. Unknown names raise an error, because a typo would silently score 0."""
    targets = config.get("target_behaviors", [])
    avoid = config.get("avoid_behaviors", [])
    unknown = [name for name in targets + avoid if name not in BEHAVIORS + SANITY_BEHAVIORS + [SANITY_ANY]]
    if len(unknown) > 0:
        raise ValueError(f"Unknown behaviors {unknown} in the experiment config. Choose from {BEHAVIORS + SANITY_BEHAVIORS + [SANITY_ANY]}")
    if is_sanity(config) and any(not name.startswith("sanity_") for name in targets + avoid):
        raise ValueError("Don't mix sanity_ behaviors with normal ones, they end the rollout in different ways.")
    if SANITY_ANY in targets:
        targets = [name for name in SANITY_BEHAVIORS if name != "sanity_no_behavior"]
    return targets, avoid


def is_sanity(config):
    return any(name.startswith("sanity_") for name in config.get("target_behaviors", []) + config.get("avoid_behaviors", []))


def validate_experiment_config(config):
    """
    Raises an error for a config whose goals can't be scored: a typo in env_setup or in the behaviors, or a goal that the
    GoalTracker can't detect from the start state, like drawer_open from a drawer that is already open or half open.
    """
    targets, avoid = get_behavior_goals(config)
    env_setup = config["env_setup"]
    validate_env_setup(env_setup)
    if is_sanity(config):
        return # the SanityTracker doesn't use these thresholds
    for joint, (idx, low, high, rising, falling) in zip(["sliding_door", "drawer", "switch", "green_light"], ARTICULATED):
        if joint not in env_setup:
            continue
        start = articulated_value(joint, env_setup[joint])
        if rising in targets + avoid and start >= low:
            raise ValueError(f"env_setup {joint} starts at {start}, but {rising} is only detected when starting below {low}")
        if falling in targets + avoid and start <= high:
            raise ValueError(f"env_setup {joint} starts at {start}, but {falling} is only detected when starting above {high}")


def label_first_interaction(states, proprio):
    """
    The labels of generate_detailed_behavior_distribtuion in analyze_calvin_touch.py: the first small motion of an episode.
    A list because the robot can displace several blocks at once, these all count in the paper.
    """
    states, proprio = np.asarray(states), np.asarray(proprio)
    for state, robot in zip(states, proprio):
        delta = state - states[0]
        displaced = ["sanity_" + name for name, idx in BLOCKS.items()
                     if np.linalg.norm(robot[0:3] - state[idx:idx + 3]) < 0.1 and np.linalg.norm(delta[idx:idx + 3]) > 0.001]
        if len(displaced) > 0:
            return displaced

        if abs(delta[0]) > 0.05:
            return ["sanity_door_left" if delta[0] > 0 else "sanity_door_right"]
        if abs(delta[1]) > 0.05:
            return ["sanity_drawer_open" if delta[1] > 0 else "sanity_drawer_close"]
        if abs(delta[3]) > 0.02:
            return ["sanity_switch_on" if delta[3] > 0 else "sanity_switch_off"]
        if abs(delta[5]) > 0.01:
            return ["sanity_button_on" if delta[5] > 0 else "sanity_button_off"]
    return ["sanity_no_behavior"]


class GoalTracker:
    """
    Follows one episode. Call update() every step with the state, it returns True once the outcome is decided:
    a target was reached or something to avoid happened. Without goals in the config it never returns True.
    """
    def __init__(self, config):
        self.targets, self.avoid = get_behavior_goals(config)
        self.start_state = None
        self.achieved = list()

    def update(self, state, proprio):
        if self.start_state is None:
            self.start_state = state
        for name in self.detect(state, proprio):
            if name not in self.achieved:
                self.achieved.append(name)
        return any(name in self.targets or name in self.avoid for name in self.achieved)

    def detect(self, state, proprio):
        detected = list()
        for idx, low, high, rising, falling in ARTICULATED: # start beyond the authors' threshold, end at COMPLETED
            if self.start_state[idx] < low and state[idx] > COMPLETED.get(rising, high):
                detected.append(rising)
            if self.start_state[idx] > high and state[idx] < COMPLETED.get(falling, low):
                detected.append(falling)
        for name, idx in BLOCKS.items():
            moved = np.linalg.norm(state[idx:idx + 3] - self.start_state[idx:idx + 3])
            if moved > BLOCK_MOVED and np.linalg.norm(proprio[0:3] - state[idx:idx + 3]) < BLOCK_ROBOT_NEAR:
                detected.append(name)
        return detected

    @property
    def behaviors(self):
        return self.achieved if len(self.achieved) > 0 else ["no_behavior"]


class SanityTracker:
    """
    Ends the episode like the paper: at the first small change of anything, whatever the goal of the config is.
    - binaries: the articulated states at the start of the episode, returned by generate_reset_state
    """
    def __init__(self, binaries):
        self.binaries = binaries
        self.states, self.proprios = list(), list()

    def update(self, state, proprio):
        self.states.append(state)
        self.proprios.append(proprio)
        return check_state_difference(self.states[0], state, proprio[0:3], self.binaries, for_display = False)

    @property
    def behaviors(self):
        return label_first_interaction(self.states, self.proprios)


def make_tracker(config, binaries):
    """Returns what ends the rollout, which depends on the behaviors in the experiment config."""
    return SanityTracker(binaries) if is_sanity(config) else GoalTracker(config)


def is_success(behaviors, targets, avoid):
    """Success means any target happened (if there are targets) and nothing to avoid happened. None if the config has no goals."""
    if len(targets) == 0 and len(avoid) == 0:
        return None
    hit_target = len(targets) == 0 or any(name in targets for name in behaviors)
    return hit_target and not any(name in avoid for name in behaviors)


def summarize_behaviors(episode_behaviors, config):
    """Takes the GoalTracker behaviors of each episode and returns the statistics that end up in the results json."""
    targets, avoid = get_behavior_goals(config)
    successes = [is_success(behaviors, targets, avoid) for behaviors in episode_behaviors]
    has_goal = len(targets) > 0 or len(avoid) > 0
    counts = dict(Counter(name for behaviors in episode_behaviors for name in behaviors))
    summary = {
        "Target_Behaviors" : targets,
        "Avoid_Behaviors" : avoid,
        "Success_Rate" : float(np.mean(successes)) if has_goal and len(successes) > 0 else None,
        "Num_Success" : int(np.sum(successes)) if has_goal else None,
        "Behavior_Counts" : counts,
        "Episode_Behaviors" : episode_behaviors,
    }
    if is_sanity(config): # the "Full distribution" that the paper plots
        summary["Full_Distribution"] = {name : count / sum(counts.values()) for name, count in counts.items()}
    return summary


def replay_episode(states, proprio, config):
    """
    Runs the GoalTracker over a recorded episode, stopping where a rollout would have stopped. states and proprio are the state
    before the first step followed by the state after every step, see evaluate_dataset.
    """
    states, proprio = np.asarray(states), np.asarray(proprio)
    if states.ndim == 3: # collapse this if there's framestacking for the policy
        states, proprio = states[:, -1], proprio[:, -1]
    if is_sanity(config):
        return label_first_interaction(states, proprio)
    tracker = GoalTracker(config)
    for state, robot in zip(states, proprio):
        if tracker.update(state, robot):
            break
    return tracker.behaviors


def episode_states(demo):
    """The start state and the state after every step, like run_dynaguide.py feeds them. Files without next_obs miss the last step."""
    states, proprio = demo["obs"]["states"][()], demo["obs"]["proprio"][()]
    if "next_obs" in demo:
        states = np.concatenate([states[:1], demo["next_obs"]["states"][()]])
        proprio = np.concatenate([proprio[:1], demo["next_obs"]["proprio"][()]])
    return states, proprio


def evaluate_dataset(hdf5_name, config):
    """Evaluates a rollout file that was saved with --dataset_path and --dataset_obs."""
    with h5py.File(hdf5_name, "r") as dataset:
        data_grp = dataset["data"]
        demos = sorted(data_grp.keys(), key = lambda x: int(x.split("_")[-1]))
        if len(demos) > 0 and "obs" not in data_grp[demos[0]]:
            raise KeyError(f"{hdf5_name} has no observations. Run the rollouts with --dataset_obs")
        episode_behaviors = [replay_episode(*episode_states(data_grp[demo]), config) for demo in demos]
    summary = summarize_behaviors(episode_behaviors, config)
    summary["Num_Episode"] = len(demos)
    return summary
