import h5py
import numpy as np
from collections import Counter
from dataclasses import dataclass, field

from core.calvin_behaviors import ARTICULATED, BEHAVIORS, ON_AVOID, detect_behaviors, episode_states
from core.calvin_utils import BLOCK_INDEX, articulated_value, validate_env_setup
from core.guidance_conditions import StateView, find_frames, state_matches

"""
Multi-goal tasks: an experiment config with a goal_sequence instead of target_behaviors, like "open the drawer, put the blue block
in, close the drawer". This file is the side of the environment, which decides whether the robot did it. The side of DynaGuide, which
guides towards one goal step after the other and decides by itself when to move on, is core/multigoal_guidance.py.

    {
        "env_setup" : {"drawer" : "closed", "blocks" : {"blue" : "table_center"}},
        "reset_poses" : "calvin_exp_configs_examples/reset_poses/....json",
        "avoid_behaviors" : ["door_left"],            optional, must not happen in any step
        "on_avoid" : "end",                           optional, what happens when one does, see below
        "switch" : {"mode" : "latent", ...},          optional, how DynaGuide moves on, see core/multigoal_guidance.py
        "goal_sequence" : [
            {"name" : "open_drawer",
             "behaviors" : ["drawer_open"],            all of these must happen during this step
             "pos_examples" : [...files or patterns...], "neg_examples" : null, "scale" : 1, "alpha" : 40},
            {"name" : "blue_in_drawer",
             "where" : {"blocks" : {"blue" : ["drawer", "released"]}},     must hold when the step completes
             "pos_examples" : [...]},
            {"name" : "close_drawer", "behaviors" : ["drawer_close"], "max_steps" : 150, "pos_examples" : [...]}
        ]
    }

What a step can ask for, everything it lists must be true at once for it to complete:
- behaviors: all must happen during the step, which starts when the step before completed. A joint behavior happens when the joint was
  on its start side at some moment of the step and reaches COMPLETED on the other side (thresholds of core/calvin_behaviors.py). So
  drawer_open and later drawer_close both count, the drawer being open at the start of the closing step. A switch that was turned on
  too early, in the step before, has to go off and on again. Blocks count from where they were at the start of the step.
  Once seen in a step, a behavior stays seen until the step completes
- any_behaviors: at least one of these must happen during the step
- where: a description of the state, in the language of find_frames in core/guidance_conditions.py (joints, blocks, tcp_near, gripper).
  For blocks: drawer, table, table_left/center/right, cabinet, cabinet_left/right, held, lifted, released, resting, lists of these,
  {"near" : [x, y, z] or color, "radius" : r} and {"on" : color}
- blocks_moved: {"blue" : 0.05}, the block is at least this far (m) from where it was at the start of the step
- avoid_behaviors: must not happen during this step, in addition to those of the config. Counted from the start of the step, like behaviors
- on_avoid: what happens when a behavior to avoid happens, for this step only (otherwise as in the config)
    "end" (default)   the rollout ends there and the episode failed
    "record"          the rollout goes on and the episode can still succeed. It is counted in Violations of the results, and
                      Num_Success_Without_Violation counts the successes where nothing to avoid happened
- max_steps: fails the episode if the step takes longer than this many env steps

A rollout ends when: all steps are done, something to avoid happened with on_avoid "end", a step took longer than its max_steps, or the
horizon of the run is over. A behavior that isn't listed to avoid never ends it.

A step needs at least one of behaviors, any_behaviors, where, blocks_moved. Later steps don't undo earlier ones: a drawer that is
opened in step 1 and closed in step 3 is a success. Steps complete strictly in order, a step can't be done before the one before it.
Several steps can complete in the same env step, if what the later one asks for already holds.

pos_examples, neg_examples, scale, alpha and switch of a step are for DynaGuide and ignored here.
"""

STEP_GOAL_KEYS = ["behaviors", "any_behaviors", "where", "blocks_moved"]
STEP_KEYS = ["name", "description", "avoid_behaviors", "on_avoid", "max_steps", "pos_examples", "neg_examples", "use_neg", "scale", "alpha", "switch"] + STEP_GOAL_KEYS
EVENT_BEHAVIORS = [name for name in BEHAVIORS if name != "no_behavior"]
JOINTS = ["sliding_door", "drawer", "switch", "green_light"] # in the order of ARTICULATED


def is_multigoal(config):
    return config is not None and "goal_sequence" in config


@dataclass
class GoalStep:
    name: str
    behaviors: list = field(default_factory = list)
    any_behaviors: list = field(default_factory = list)
    where: dict = field(default_factory = dict)
    blocks_moved: dict = field(default_factory = dict)
    avoid_behaviors: list = field(default_factory = list)
    max_steps: int = None
    on_avoid: str = "end"
    raw: dict = field(default_factory = dict) # the whole entry of the config, with the guidance settings


def avoid_mode(value, label):
    if value not in ON_AVOID:
        raise ValueError(f"{label} on_avoid: choose from {ON_AVOID}, got {value}")
    return value


def parse_goal_sequence(config):
    """Reads and checks the goal steps of a config. Raises an error for anything that wouldn't do what it says."""
    sequence = config.get("goal_sequence")
    if not isinstance(sequence, list) or len(sequence) == 0:
        raise ValueError("goal_sequence must be a list with at least one goal step")
    if "target_behaviors" in config:
        raise ValueError("A config with goal_sequence can't have target_behaviors, put the goals into the steps")
    default_on_avoid = avoid_mode(config.get("on_avoid", "end"), "config")
    steps = list()
    for number, entry in enumerate(sequence):
        label = f"goal_sequence step {number}"
        if not isinstance(entry, dict):
            raise ValueError(f"{label}: expected a dict, got {entry}")
        unknown = [key for key in entry if key not in STEP_KEYS]
        if len(unknown) > 0:
            raise ValueError(f"{label}: unknown keys {unknown}, choose from {STEP_KEYS}")
        if not any(entry.get(key) for key in STEP_GOAL_KEYS):
            raise ValueError(f"{label}: needs at least one of {STEP_GOAL_KEYS}, otherwise it would be done at once")
        name = entry.get("name", f"step_{number}")
        for key in ["behaviors", "any_behaviors", "avoid_behaviors"]:
            names = entry.get(key, [])
            if not isinstance(names, list) or any(item not in EVENT_BEHAVIORS for item in names):
                raise ValueError(f"{label} ({name}) {key}: expected a list of {EVENT_BEHAVIORS}, got {names}")
        where = entry.get("where", {})
        find_frames(StateView(np.zeros(24), np.zeros(15)), where) # a dry run, raises an error for unknown keys and words
        moved = entry.get("blocks_moved", {})
        if not isinstance(moved, dict) or any(color not in BLOCK_INDEX or not isinstance(distance, (int, float)) or distance <= 0
                                              for color, distance in moved.items()):
            raise ValueError(f"{label} ({name}) blocks_moved: expected {{color : distance in m}} with colors {list(BLOCK_INDEX)}, got {moved}")
        max_steps = entry.get("max_steps")
        if max_steps is not None and (not isinstance(max_steps, int) or max_steps <= 0):
            raise ValueError(f"{label} ({name}) max_steps: expected a positive integer, got {max_steps}")
        steps.append(GoalStep(name, list(entry.get("behaviors", [])), list(entry.get("any_behaviors", [])), dict(where), dict(moved),
                              list(entry.get("avoid_behaviors", [])), max_steps,
                              avoid_mode(entry.get("on_avoid", default_on_avoid), f"{label} ({name})"), dict(entry)))
    names = [step.name for step in steps]
    if len(set(names)) != len(names):
        raise ValueError(f"Two goal steps have the same name: {names}")
    avoid = config.get("avoid_behaviors", [])
    if any(item not in EVENT_BEHAVIORS for item in avoid):
        raise ValueError(f"avoid_behaviors: expected a list of {EVENT_BEHAVIORS}, got {avoid}")
    for step in steps:
        both = sorted((set(step.behaviors) | set(step.any_behaviors)) & (set(step.avoid_behaviors) | set(avoid)))
        if len(both) > 0:
            raise ValueError(f"goal step {step.name}: {both} is to be reached and to be avoided. Put avoid_behaviors into the steps where they apply")
    return steps


def validate_multigoal_config(config):
    """
    Raises an error for a goal_sequence that can't be scored: typos, or a joint behavior that can't happen because of what the steps
    before it leave, like drawer_close as the first step when env_setup starts with a closed drawer. Joints left random by env_setup
    aren't checked until a step fixes them.
    """
    steps = parse_goal_sequence(config)
    env_setup = config["env_setup"]
    validate_env_setup(env_setup)
    side = dict() # joint -> "low" / "high" as far as we know it
    for joint, (idx, low, high, _, _) in zip(JOINTS, ARTICULATED):
        if joint in env_setup:
            start = articulated_value(joint, env_setup[joint])
            side[joint] = "low" if start < low else "high" if start > high else "middle"
    for step in steps:
        for joint, (idx, low, high, rising, falling) in zip(JOINTS, ARTICULATED):
            for name in step.behaviors:
                if name == rising and side.get(joint, "low") != "low":
                    raise ValueError(f"goal step {step.name}: {rising} needs the {joint} below {low} at the start of the step, "
                                     f"but the steps before leave it {side[joint]}")
                if name == falling and side.get(joint, "high") != "high":
                    raise ValueError(f"goal step {step.name}: {falling} needs the {joint} above {high} at the start of the step, "
                                     f"but the steps before leave it {side[joint]}")
            if rising in step.behaviors:
                side[joint] = "high"
            if falling in step.behaviors:
                side[joint] = "low"
            if isinstance(step.where.get(joint), str): # a word of find_frames, like "open"
                side[joint] = "low" if step.where[joint] in ["closed", "right", "off"] else "high"
    return steps


@dataclass
class StepResult:
    name: str
    started_at: int
    completed_at: int = None
    events: list = field(default_factory = list) # (env step, behavior) seen during the step


class SequentialGoalTracker:
    """
    Follows one episode of a goal_sequence. Call update() with the state before the first step and after every step. It returns True
    once the episode is decided: all steps done, something to avoid happened, or a step ran out of max_steps.
    Same interface as GoalTracker in core/calvin_behaviors.py, so run_dynaguide.py can use either.
    """
    def __init__(self, config):
        self.steps = parse_goal_sequence(config)
        self.global_avoid = config.get("avoid_behaviors", [])
        self.t = -1 # env steps since the start, the start state is t = 0
        self.current = 0
        self.reference = None # state and proprio at the start of the current step
        self.seen = set() # behaviors of the current step
        self.results = list()
        self.achieved = list() # every behavior of the episode, in order and each once, for the Behavior_Counts
        self.violations = list() # (env step, behavior, goal step) to avoid that happened
        self.failure = None

    @property
    def done(self):
        return self.success or self.failure is not None

    @property
    def success(self):
        return self.current >= len(self.steps)

    @property
    def steps_completed(self):
        return min(self.current, len(self.steps))

    def _start_step(self, state, proprio):
        self.reference = (np.array(state, dtype = np.float64), np.array(proprio, dtype = np.float64))
        self.lowest, self.highest = self.reference[0].copy(), self.reference[0].copy() # of each value during the step
        self.seen = set()
        if self.current < len(self.steps):
            self.results.append(StepResult(self.steps[self.current].name, self.t))

    def _complete(self, step, state, proprio):
        if not set(step.behaviors) <= self.seen:
            return False
        if len(step.any_behaviors) > 0 and len(self.seen & set(step.any_behaviors)) == 0:
            return False
        if len(step.where) > 0 and not state_matches(state, proprio, step.where):
            return False
        for color, distance in step.blocks_moved.items():
            idx = BLOCK_INDEX[color]
            if np.linalg.norm(state[idx:idx + 3] - self.reference[0][idx:idx + 3]) < distance:
                return False
        return True

    def update(self, state, proprio):
        if self.done:
            return True
        self.t += 1
        state, proprio = np.asarray(state, dtype = np.float64), np.asarray(proprio, dtype = np.float64)
        if self.reference is None:
            self._start_step(state, proprio)
        while not self.done:
            step = self.steps[self.current]
            self.lowest, self.highest = np.minimum(self.lowest, state), np.maximum(self.highest, state)
            for name in detect_behaviors(self.reference[0], state, proprio, self.lowest, self.highest):
                first = name not in self.seen
                if first:
                    self.seen.add(name)
                    self.results[-1].events.append((self.t, name))
                if name not in self.achieved:
                    self.achieved.append(name)
                if first and (name in self.global_avoid or name in step.avoid_behaviors):
                    self.violations.append((self.t, name, step.name))
                    if step.on_avoid == "end":
                        self.failure = f"{name} during step {step.name}"
                        return True
            if step.max_steps is not None and self.t - self.results[-1].started_at > step.max_steps:
                self.failure = f"step {step.name} took more than {step.max_steps} steps"
                return True
            if not self._complete(step, state, proprio):
                break
            self.results[-1].completed_at = self.t
            self.current += 1
            self._start_step(state, proprio) # the next step is measured from here
        return self.done

    @property
    def behaviors(self):
        return self.achieved if len(self.achieved) > 0 else ["no_behavior"]

    def summary(self):
        """What run_dynaguide.py writes about this episode."""
        return {"success" : self.success, "steps_completed" : self.steps_completed, "failure" : self.failure, "env_steps" : self.t,
                "behaviors" : self.behaviors, "violations" : [list(violation) for violation in self.violations],
                "steps" : [{"name" : result.name, "started_at" : result.started_at, "completed_at" : result.completed_at,
                            "events" : [[t, name] for t, name in result.events]} for result in self.results]}


def summarize_sequence(episodes, config, switch_logs = None):
    """
    The statistics of a multi-goal run for the results json.
    - episodes: SequentialGoalTracker.summary() of each episode
    - switch_logs: MultiGoalGuidance.log() of each episode, to compare when DynaGuide moved on with when the env saw the step done
    """
    steps = parse_goal_sequence(config)
    n = len(episodes)
    successes = [episode["success"] for episode in episodes]
    completed = [episode["steps_completed"] for episode in episodes]
    counts = dict(Counter(name for episode in episodes for name in episode["behaviors"]))
    summary = {
        "Goal_Sequence" : [step.name for step in steps],
        "Success_Rate" : float(np.mean(successes)) if n > 0 else None,
        "Num_Success" : int(np.sum(successes)),
        "Num_Success_Without_Violation" : sum(episode["success"] and len(episode.get("violations", [])) == 0 for episode in episodes),
        "Violations" : dict(Counter(f"{name} during step {step}" for episode in episodes for _, name, step in episode.get("violations", []))),
        "Mean_Steps_Completed" : float(np.mean(completed)) if n > 0 else None,
        "Step_Reach_Rate" : {step.name : float(np.mean([c > k for c in completed])) if n > 0 else None for k, step in enumerate(steps)},
        "Failures" : dict(Counter(episode["failure"] for episode in episodes if episode["failure"] is not None)),
        "Behavior_Counts" : counts,
        "Episode_Behaviors" : [episode["behaviors"] for episode in episodes],
        "Episode_Goal_Steps" : episodes,
    }
    if switch_logs is not None:
        summary["Switching"] = switching_statistics(episodes, switch_logs, steps)
        summary["Episode_Switches"] = switch_logs
    return summary


def switching_statistics(episodes, switch_logs, steps):
    """
    For each step but the last: how often DynaGuide moved on from it, how often before the env saw it done (too early), and the delay
    from the env seeing it done to the switch. The env sees a step done at completed_at, DynaGuide moves on at the switch's step.
    """
    statistics = dict()
    for k, step in enumerate(steps[:-1]):
        switched, early, never, delays = 0, 0, 0, list()
        for episode, log in zip(episodes, switch_logs):
            env_done = episode["steps"][k]["completed_at"] if k < len(episode["steps"]) else None
            switch = next((entry["t"] for entry in log["switches"] if entry["from_step"] == k), None)
            if switch is None:
                never += int(env_done is not None) # the env saw it done, DynaGuide didn't move on
                continue
            switched += 1
            if env_done is None or switch < env_done:
                early += 1
            else:
                delays.append(switch - env_done)
        statistics[step.name] = {"switched" : switched, "too_early" : early, "missed" : never,
                                 "mean_delay" : float(np.mean(delays)) if len(delays) > 0 else None,
                                 "max_delay" : int(np.max(delays)) if len(delays) > 0 else None}
    return statistics


def replay_sequence(states, proprio, config):
    """Runs the SequentialGoalTracker over a recorded episode, like a rollout would. Returns its summary()."""
    states, proprio = np.asarray(states), np.asarray(proprio)
    if states.ndim == 3: # frame stacking of the policy
        states, proprio = states[:, -1], proprio[:, -1]
    tracker = SequentialGoalTracker(config)
    for state, robot in zip(states, proprio):
        if tracker.update(state, robot):
            break
    return tracker.summary()


def evaluate_sequence_dataset(hdf5_name, config):
    """Scores a rollout file that was saved with --dataset_obs against a goal_sequence, see evaluate_behaviors.py."""
    with h5py.File(hdf5_name, "r") as dataset:
        data_grp = dataset["data"]
        demos = sorted(data_grp.keys(), key = lambda x: int(x.split("_")[-1]))
        if len(demos) > 0 and "obs" not in data_grp[demos[0]]:
            raise KeyError(f"{hdf5_name} has no observations. Run the rollouts with --dataset_obs")
        episodes = [replay_sequence(*episode_states(data_grp[demo]), config) for demo in demos]
    summary = summarize_sequence(episodes, config)
    summary["Num_Episode"] = len(demos)
    return summary
