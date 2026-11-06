import h5py
import numpy as np
from collections import Counter

"""
Goal conditions for CALVIN rollouts. The environment has no success signal of its own, so we detect what the robot achieved
from the states and compare this to the target_behaviors / avoid_behaviors of the experiment config. run_dynaguide.py only
asks this file whether the goal is decided, it doesn't look at the state itself.

The thresholds are the end states from the segmentation in data_processing_calvin/calvin_to_labeled_hdf5.py, which define what
counts as a behavior in the guidance conditions. They are stricter than analyze_calvin_touch.py, which looks for the first small motion.
"""

BEHAVIORS = ["button_on", "button_off", "switch_on", "switch_off", "drawer_open", "drawer_close", "door_left", "door_right",
             "red_displace", "blue_displace", "pink_displace", "no_behavior"]

# state index, low, high, name when going from below low to above high, name when going from above high to below low
# the index is the one in the lowdim state, see generate_reset_state in calvin_utils.py
ARTICULATED = [(0, 0.05, 0.25, "door_left", "door_right"),
               (1, 0.05, 0.10, "drawer_open", "drawer_close"),
               (3, 0.01, 0.07, "switch_on", "switch_off"),
               (5, 0.2, 0.8, "button_on", "button_off")]

BLOCKS = {"red_displace" : 6, "blue_displace" : 12, "pink_displace" : 18} # index of the block position
BLOCK_MOVED = 0.01
BLOCK_ROBOT_NEAR = 0.1 # so we don't count a block that moves with the drawer


def get_behavior_goals(config):
    """Reads the goals of an experiment config. Unknown names raise an error, because a typo would silently score 0."""
    targets = config.get("target_behaviors", [])
    avoid = config.get("avoid_behaviors", [])
    unknown = [name for name in targets + avoid if name not in BEHAVIORS]
    if len(unknown) > 0:
        raise ValueError(f"Unknown behaviors {unknown} in the experiment config. Choose from {BEHAVIORS}")
    return targets, avoid


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
        for idx, low, high, rising, falling in ARTICULATED:
            if self.start_state[idx] < low and state[idx] > high:
                detected.append(rising)
            if self.start_state[idx] > high and state[idx] < low:
                detected.append(falling)
        for name, idx in BLOCKS.items():
            moved = np.linalg.norm(state[idx:idx + 3] - self.start_state[idx:idx + 3])
            if moved > BLOCK_MOVED and np.linalg.norm(proprio[0:3] - state[idx:idx + 3]) < BLOCK_ROBOT_NEAR:
                detected.append(name)
        return detected

    @property
    def behaviors(self):
        return self.achieved if len(self.achieved) > 0 else ["no_behavior"]


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
    return {
        "Target_Behaviors" : targets,
        "Avoid_Behaviors" : avoid,
        "Success_Rate" : float(np.mean(successes)) if has_goal and len(successes) > 0 else None,
        "Num_Success" : int(np.sum(successes)) if has_goal else None,
        "Behavior_Counts" : dict(Counter(name for behaviors in episode_behaviors for name in behaviors)),
        "Episode_Behaviors" : episode_behaviors,
    }


def replay_episode(states, proprio, config):
    """Runs the GoalTracker over a recorded episode, stopping where a rollout would have stopped."""
    states, proprio = np.asarray(states), np.asarray(proprio)
    if states.ndim == 3: # collapse this if there's framestacking for the policy
        states, proprio = states[:, -1], proprio[:, -1]
    tracker = GoalTracker(config)
    for state, robot in zip(states, proprio):
        if tracker.update(state, robot):
            break
    return tracker.behaviors


def evaluate_dataset(hdf5_name, config):
    """Evaluates a rollout file that was saved with --dataset_path and --dataset_obs."""
    with h5py.File(hdf5_name, "r") as dataset:
        data_grp = dataset["data"]
        demos = sorted(data_grp.keys(), key = lambda x: int(x.split("_")[-1]))
        if len(demos) > 0 and "obs" not in data_grp[demos[0]]:
            raise KeyError(f"{hdf5_name} has no observations. Run the rollouts with --dataset_obs")
        episode_behaviors = [replay_episode(data_grp[demo]["obs"]["states"][()], data_grp[demo]["obs"]["proprio"][()], config) for demo in demos]
    summary = summarize_behaviors(episode_behaviors, config)
    summary["Num_Episode"] = len(demos)
    return summary
