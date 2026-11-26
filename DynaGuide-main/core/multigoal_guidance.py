import numpy as np
import torch

from core.dynaguide import calculate_classifier_guidance, prepare_np

"""
Multi-goal guidance: DynaGuide guides towards the conditions of one goal step at a time, and decides by itself when a step is done and
it moves on to the next. The environment's side (did the robot really do it) is core/goal_sequence.py. See "Multigoal guidance.ipynb"
for the design, the calibration of the switch, and the tests.

The switch, set by "switch" in the experiment config (for all steps) and in each goal step (overrides it for that step):

    mode            "latent" (default): DynaGuide decides from its own embedding of the camera image, see below
                    "oracle": moves on delay env steps after the SequentialGoalTracker of the env sees the step done. For comparison
                    only, the robot doesn't know the simulator state. With the oracle, a failure is the guidance's fault, not the switch's
    criterion       "patches" (default for a step with before_examples since 2026-11-26), see below
                    "change": asks whether the object changed as in the conditions. before_examples
                    show the same scene with the same robot as each goal condition, but with the step's change not done yet: drawer still
                    closed, LED still off. The difference of the two images is the change, and the score says how much of it the current
                    image shows: -1 as before, +1 as in the goal condition (change_score below). The arm is the same in both images, so
                    it isn't part of the change, wherever it is. The score has no unit, so one margin fits all goals.
                    Done when the score is above margin and stopped rising, i.e. no new maximum (by more than tolerance) for patience
                    checks: so not halfway while the drawer is still moving. A step whose goal already holds at its start is done at once.
                    make_task.py renders the before_examples of every step that changes a joint
                    "patches" (needs before_examples too): the same score, but only on the patches the step changes, those that differ
                    between a goal condition and its before_example by more than changed (0.3) times the strongest one. The best
                    fitting pair counts, not the average: for a block that is let go, one condition with the right place is enough.
                    Done as with change, with the margin 0.6 (0.5 until 2026-11-26: 14% false switches on recorded release rollouts,
                    0.6 one in 278, joints alike from 0.5 to 0.7). For a release step make_task.py writes goal_examples (the block at a grid
                    of places) and before_examples (the same without the block) for the switch alone, the guidance keeps its own
                    conditions, and default_criterion patches: what the step uses when no criterion is given
                    "peak" (default for a step without before_examples): the older switch, on the whole image. Uses the contrast score
                    c = d_contrast - d_goal: how much more the current image looks like the goal conditions than like the contrast
                    conditions (contrast_examples of the step, by default the goal conditions of the step before). Done when c has risen
                    above margin and stopped rising, as above. Only the peak after the last check with c <= 0 counts, and there must be
                    one: a step that starts out looking done isn't skipped. The arm is in both distances and doesn't cancel out if it
                    stands elsewhere than in the contrast conditions, and the level of c differs a lot between behaviors (drawer ~200,
                    door ~100 to 190, switch ~40 to 120), so one margin doesn't fit all of them
                    "contrast": done when c > margin for patience checks in a row
                    "relative": d_goal < ratio * spread of the goal conditions, where the spread is the median distance of a condition
                              to its nearest other one. Needs no contrast, but robot pose and blocks count in full
                    "threshold": d_goal < threshold, in the units of the embedding (L2 of the flattened patches)
                    A step without before_examples uses "peak" instead of "change" and "patches", without contrast conditions either (the first step)
                    "relative" instead of "peak" and "contrast"
    margin, tolerance, patience     see the criteria. Defaults for change: 0.2, 0.05 and 10, for patches: 0.6, 0.05 and 10, for the
                    others: 30, 5 and 5 (margin and tolerance in the units of the embedding)
    changed         for patches: how strongly a patch must change to count, as a part of the strongest one (default 0.3)
    ratio, threshold    the numbers of relative and threshold
    embedding       what is compared, "observed" (default) or "predicted" (not for change and patches)
                    observed: the DINO patches of the current camera image against those of the last frames of the step's conditions.
                              Both are encoder outputs of real images, no prediction error, and they don't depend on the action
                    predicted: the dynamics model's prediction for the current image with the last action held, against the
                              condition embeddings the guidance uses
    delay           for the oracle: env steps it keeps guiding to the old step after the env saw it done (default 10), so a joint that is
                    still moving gets to its end. The latent switches wait about as long (patience 10). Runs before 2026-11-26 had 0
    min_steps       env steps a step is guided at least, before the switch is checked (default 10)
    check_every     check every this many env steps (default 1)
    replan_on_switch    throw away the queued actions of the old goal and plan with the new one right away (default true)

d_goal and d_contrast are soft minimums over the conditions, -alpha * logsumexp(-d / alpha), with the alpha of the step. That is the
distance the guidance itself pulls on, close to the distance of the nearest condition.

dynaguide_scripts/test_goal_switch.py replays the switch over recorded rollouts, calibrate_goal_switching.py measures the older criteria.
The numbers behind the defaults are in the notebook.
"""

SWITCH_DEFAULTS = {"mode" : "latent", "embedding" : "observed", "criterion" : "peak", "margin" : 30.0, "tolerance" : 5.0, "ratio" : 1.0,
                   "threshold" : None, "patience" : 5, "min_steps" : 10, "check_every" : 1, "replan_on_switch" : True, "changed" : 0.3,
                   "delay" : 10}
CHANGE_DEFAULTS = {"criterion" : "change", "margin" : 0.2, "tolerance" : 0.05, "patience" : 10} # what differs for the change criterion
PATCHES_DEFAULTS = {"criterion" : "patches", "margin" : 0.6, "tolerance" : 0.05, "patience" : 10} # the same for the patches criterion
PAIR_DEFAULTS = {"change" : CHANGE_DEFAULTS, "patches" : PATCHES_DEFAULTS} # the criteria that compare with the before_examples
CRITERION_NUMBERS = ["margin", "tolerance", "patience"] # mean something else for these than for the others
PATCH_DIM = 384 # numbers per DINO patch, the embeddings are the patches one after the other
SWITCH_KEYS = list(SWITCH_DEFAULTS) + ["contrast_examples", "before_examples", "goal_examples", "default_criterion"]
CRITERIA = ["change", "patches", "peak", "contrast", "relative", "threshold"]


def switch_settings(config, step_entry):
    """The switch settings of a step: the defaults of its criterion, then the config's switch, then the step's."""
    given = dict()
    sources = [config.get("switch", {}), step_entry.get("switch", {})]
    for source in sources:
        unknown = [key for key in source if key not in SWITCH_KEYS]
        if len(unknown) > 0:
            raise ValueError(f"Unknown switch settings {unknown}, choose from {SWITCH_KEYS}")
        given.update(source)
    has_before = given.get("before_examples") is not None
    criterion = given.get("criterion", given.get("default_criterion", "patches") if has_before and given.get("embedding", "observed") == "observed" else "peak")
    if criterion not in CRITERIA:
        raise ValueError(f"switch criterion must be one of {CRITERIA}, got {criterion}")
    if criterion in PAIR_DEFAULTS and not has_before:
        if "criterion" in sources[1]:
            raise ValueError(f"switch criterion {criterion} needs before_examples")
        given = {key : value for key, value in given.items() if key not in CRITERION_NUMBERS or key in sources[1]} # the config's are for the pairs
        criterion = "peak"
    settings = {**SWITCH_DEFAULTS, **PAIR_DEFAULTS.get(criterion, {}), **given, "criterion" : criterion}
    if settings["mode"] not in ["latent", "oracle"]:
        raise ValueError(f"switch mode must be latent or oracle, got {settings['mode']}")
    if settings["embedding"] not in ["observed", "predicted"]:
        raise ValueError(f"switch embedding must be observed or predicted, got {settings['embedding']}")
    if criterion == "threshold" and settings["threshold"] is None:
        raise ValueError("switch criterion threshold needs a threshold")
    if criterion in PAIR_DEFAULTS and settings["embedding"] != "observed":
        raise ValueError(f"switch criterion {criterion} needs the observed embedding")
    if not 0 <= settings["changed"] < 1:
        raise ValueError(f"changed is a part of the strongest patch, 0 to 1, got {settings['changed']}")
    if criterion in PAIR_DEFAULTS and not -1 < settings["margin"] < 1:
        raise ValueError(f"the score of the {criterion} criterion goes from -1 (before) to 1 (done), a margin of {settings['margin']} makes no sense")
    return settings

# --- embeddings -------------------------------------------------------------------------------------------------------------------------

def hold_chunk(action, length):
    """An action chunk that repeats one action, like the padding of the conditions (pad_mode repeat)."""
    return np.repeat(np.asarray(action, dtype = np.float32)[None], length, axis = 0)


def condition_last_frames(dataset):
    """The last frame of each condition (state dict with the image as CHW 0-255, the action chunk), as calculate_classifier_guidance uses it."""
    frames, idx = list(), 0
    for length in dataset.lengths_list:
        idx += length
        state, actions, _ = dataset.get_labeled_item(idx - 1)
        frames.append((state, actions))
    return frames


def embed_states(model, states, actions = None, device = "cpu", batch_size = 16):
    """
    Flattened embeddings [N, D] of a list of state dicts (camera image CHW 0-255, proprio). Without actions, the DINO patches of the image
    (observed). With a list of action chunks, the dynamics model's prediction (predicted), as the guidance computes it.
    """
    outputs = list()
    for start in range(0, len(states), batch_size):
        batch = states[start:start + batch_size]
        state = {key : torch.stack([prepare_np(s[key], device) for s in batch]) for key in batch[0]}
        with torch.no_grad():
            if actions is None:
                embedding = model.state_embedding(state)
            else:
                chunk = torch.stack([prepare_np(a, device) for a in actions[start:start + batch_size]])
                embedding = model.state_action_embedding(state, chunk)
        outputs.append(embedding.flatten(start_dim = 1))
    return torch.cat(outputs, dim = 0)


def soft_min_distance(embeddings, query, alpha):
    """-alpha * logsumexp(-d / alpha) over the conditions, for each query [Q]. Goes to the nearest distance as alpha gets small."""
    return soft_min_rows(torch.cdist(query, embeddings, p = 2.0), alpha)


def soft_min_rows(distances, alpha):
    """The soft minimum of each row of a distance matrix [Q x K]."""
    return -alpha * torch.logsumexp(-distances / alpha, dim = 1)


def spread(embeddings):
    """Median distance of each condition to its nearest other one, the size of a condition set in the embedding."""
    if len(embeddings) < 2:
        return None
    distances = torch.cdist(embeddings, embeddings, p = 2.0)
    distances.fill_diagonal_(float("inf"))
    return float(torch.median(distances.min(dim = 1).values))


def change_score(goal, before, query):
    """
    How much of the change from the before images to the goal images each query shows: -1 as before, +1 as in the goal conditions.
    - goal, before: embeddings [K, D], goal[k] and before[k] show the same scene and robot, apart from the change of the step
    - query: embeddings [Q, D]
    For one pair this is the query projected on the line from before to goal. What doesn't lie along this line, like another robot pose,
    adds nothing, and a patch counts as much as the change moves it. Over the pairs it is the average weighted by how far they
    are apart, so a pair that barely shows the change (the arm hides the object) barely counts.
    """
    direction, middle = goal - before, (goal + before) / 2
    along = torch.stack([2 * ((q - middle) * direction).sum(dim = 1) for q in query]) # [Q, K], squared distance to before minus to the goal
    return along.sum(dim = 1) / (direction * direction).sum()


def patches_score(goal, before, query, changed = 0.3):
    """
    Like change_score, but only on the patches that change, and the best fitting pair counts: -1 as before, +1 as in the goal condition.
    - goal, before: embeddings [K, D], goal[k] and before[k] show the same scene and robot, apart from the change of the step
    - query: embeddings [Q, D]
    - changed: a patch of a pair counts if it differs by more than this part of the pair's strongest one
    """
    goal, before = goal.reshape(len(goal), -1, PATCH_DIM), before.reshape(len(before), -1, PATCH_DIM) # [K, patches, numbers]
    difference = (goal - before).norm(dim = 2)
    counts = (difference > changed * difference.max(dim = 1, keepdim = True).values).to(goal.dtype) # [K, patches]
    size = ((difference ** 2) * counts).sum(dim = 1)
    scores = list()
    for q in query.reshape(len(query), -1, PATCH_DIM):
        to_goal, to_before = [(((q - images) ** 2).sum(dim = 2) * counts).sum(dim = 1) for images in [goal, before]] # [K]
        score = (to_before - to_goal) / size
        scores.append(torch.where(size > 0, score, torch.full_like(score, -1.0)).max()) # a pair without a change says nothing
    return torch.stack(scores)


PAIR_SCORES = {"change" : lambda goal, before, query, settings : change_score(goal, before, query),
               "patches" : lambda goal, before, query, settings : patches_score(goal, before, query, settings["changed"])}


def observation_state(obs, main_camera, proprio_key = "proprio"):
    """The model's input for an observation of the env: the last frame of the frame stack, image as CHW 0-255."""
    image, proprio = np.asarray(obs[main_camera]), np.asarray(obs[proprio_key])
    if image.ndim == 4:
        image, proprio = image[-1], proprio[-1]
    return {main_camera : image * 255.0, proprio_key : proprio}


def check_pairs(name, goal_dataset, before_dataset):
    """The before_examples must be the twins of the goal conditions: as many, and each with the robot of its goal condition."""
    goal, before = condition_last_frames(goal_dataset), condition_last_frames(before_dataset)
    if len(goal) != len(before):
        raise ValueError(f"Goal step {name}: {len(goal)} goal conditions but {len(before)} before_examples, each goal condition needs its own")
    for number, ((goal_state, _), (before_state, _)) in enumerate(zip(goal, before)):
        if not np.allclose(goal_state["proprio"], before_state["proprio"], atol = 1e-4):
            raise ValueError(f"Goal step {name}: condition {number} and its before_example don't have the same robot pose")

# --- deciding that a step is done -------------------------------------------------------------------------------------------------------

class SwitchCriterion:
    """
    Decides from one check after the other whether a goal step is done, see the criteria at the top. Works on the numbers only, so the
    calibration can run it over recorded distances exactly as a rollout does.
    """
    def __init__(self, settings, criterion, goal_spread = None):
        self.settings = settings
        self.criterion = criterion
        self.goal_spread = goal_spread
        if criterion == "relative" and goal_spread is None:
            raise ValueError("The relative switch needs at least two goal conditions, or give contrast_examples")
        self.reset()

    def reset(self):
        self.streak = 0 # checks in a row that look done (contrast, relative, threshold)
        self.best = -float("inf") # highest score of the step so far (change, patches, peak)
        self.since_best = 0
        self.crossed = False # the contrast score was <= 0 during the step

    def score(self, d_goal, d_contrast):
        """What the trace keeps: the contrast score for peak and contrast, else d_goal minus where it would be done."""
        if self.criterion in ["peak", "contrast"]:
            return d_contrast - d_goal
        if self.criterion == "relative":
            return d_goal - self.settings["ratio"] * self.goal_spread
        return d_goal - self.settings["threshold"]

    def update(self, d_goal, d_contrast):
        """One check of the criteria that work on distances. Returns True when the step is done."""
        return self.update_score(self.score(d_goal, d_contrast))

    def update_score(self, score):
        """One check. Returns True when the step is done."""
        if self.criterion in PAIR_DEFAULTS:
            if score <= 0 or score > self.best + self.settings["tolerance"]: # not there yet, or still rising
                self.best, self.since_best = score, 0
            else:
                self.since_best += 1
            return score > self.settings["margin"] and self.since_best >= self.settings["patience"]
        if self.criterion == "peak":
            margin, tolerance = self.settings["margin"], self.settings["tolerance"]
            if score <= 0: # looks like the contrast: a peak only counts after this
                self.crossed, self.best, self.since_best = True, score, 0
            elif score > self.best + tolerance:
                self.best, self.since_best = score, 0
            else:
                self.since_best += 1
            return self.crossed and score > margin and self.best > margin and self.since_best >= self.settings["patience"]
        looks_done = score > self.settings["margin"] if self.criterion == "contrast" else score < 0
        self.streak = self.streak + 1 if looks_done else 0
        return self.streak >= self.settings["patience"]

# --- the guidance -----------------------------------------------------------------------------------------------------------------------

class GoalStepGuidance:
    """What DynaGuide holds for one goal step: its guidance function, and the embeddings to decide when the step is done."""
    def __init__(self, name, guidance, good_embeddings, bad_embeddings, alpha, switch, goal_embeddings, contrast_embeddings,
                 before_embeddings = None):
        self.name = name
        self.guidance = guidance
        self.good_embeddings = good_embeddings # what the guidance pulls towards, also for the video collage
        self.bad_embeddings = bad_embeddings
        self.alpha = alpha
        self.switch = switch
        self.goal_embeddings = goal_embeddings # in the space of switch["embedding"]
        self.contrast_embeddings = contrast_embeddings
        self.before_embeddings = before_embeddings # before_embeddings[k] belongs to goal_embeddings[k]
        self.criterion = None
        if switch["mode"] == "latent" and goal_embeddings is not None:
            criterion = switch["criterion"]
            if criterion in PAIR_DEFAULTS and len(before_embeddings) != len(goal_embeddings):
                raise ValueError(f"Goal step {name}: {len(goal_embeddings)} goal conditions but {len(before_embeddings)} before_examples, "
                                 "each goal condition needs its own")
            if criterion in ["peak", "contrast"] and contrast_embeddings is None:
                print(f"Goal step {name}: no contrast conditions, the switch uses the relative criterion instead of {criterion}")
                criterion = "relative"
            self.criterion = SwitchCriterion(switch, criterion, spread(goal_embeddings))

    def distances(self, query):
        """d_goal and d_contrast (None without contrast conditions) of one query embedding [1, D]."""
        d_goal = float(soft_min_distance(self.goal_embeddings, query, self.alpha)[0])
        d_contrast = None if self.contrast_embeddings is None else float(soft_min_distance(self.contrast_embeddings, query, self.alpha)[0])
        return d_goal, d_contrast

    def check(self, query):
        """One check of the switch for one query embedding [1, D]. Returns (done, score, d_goal, d_contrast)."""
        d_goal, d_contrast = self.distances(query)
        if self.criterion.criterion in PAIR_DEFAULTS:
            score = float(PAIR_SCORES[self.criterion.criterion](self.goal_embeddings, self.before_embeddings, query, self.switch)[0])
        else:
            score = self.criterion.score(d_goal, d_contrast)
        return self.criterion.update_score(score), score, d_goal, d_contrast


class MultiGoalGuidance:
    """
    Guidance over a sequence of goal steps. Pass it to the policy as the guidance function: it guides with the current step. Call observe()
    after every env step, which decides whether the current step is done and moves on. reset() before each episode.
    """
    def __init__(self, model, steps, main_camera, device, action_chunk_length = 16):
        self.model = model
        self.steps = steps
        self.main_camera = main_camera
        self.device = device
        self.action_chunk_length = action_chunk_length
        self.reset()

    @classmethod
    def build(cls, model, step_inputs, main_camera, device, max_examples = None, action_chunk_length = 16):
        """
        - step_inputs: per step a dict with name, good_dataset, bad_dataset, contrast_dataset, before_dataset and goal_dataset (or None),
          scale, alpha, switch (settings). goal_dataset: the images the switch takes instead of the conditions, see goal_examples
        The contrast conditions default to the goal conditions of the step before.
        """
        steps = list()
        for number, inputs in enumerate(step_inputs):
            print(f"Goal step {number}: {inputs['name']}")
            guidance, good, bad = calculate_classifier_guidance(model, good_dataset = inputs["good_dataset"], main_camera = main_camera,
                                                                scale = inputs["scale"], bad_dataset = inputs["bad_dataset"],
                                                                alpha = inputs["alpha"], max_examples = max_examples, device = device)
            switch = inputs["switch"]
            goal, contrast, before = None, None, None
            if switch["mode"] == "latent" and number < len(step_inputs) - 1: # the last step is ended by the env
                goal = cls._switch_embeddings(model, inputs["good_dataset"], switch["embedding"], device, max_examples)
                if switch["criterion"] in PAIR_DEFAULTS:
                    own = inputs.get("goal_dataset")
                    if own is not None:
                        goal = cls._switch_embeddings(model, own, switch["embedding"], device, max_examples)
                    check_pairs(inputs["name"], inputs["good_dataset"] if own is None else own, inputs["before_dataset"])
                    before = cls._switch_embeddings(model, inputs["before_dataset"], switch["embedding"], device, max_examples)
                elif inputs.get("contrast_dataset") is not None:
                    contrast = cls._switch_embeddings(model, inputs["contrast_dataset"], switch["embedding"], device, max_examples)
                elif number > 0:
                    previous = steps[-1].goal_embeddings # None if the step before uses the oracle
                    contrast = previous if previous is not None and steps[-1].switch["embedding"] == switch["embedding"] else \
                        cls._switch_embeddings(model, step_inputs[number - 1]["good_dataset"], switch["embedding"], device, max_examples)
            steps.append(GoalStepGuidance(inputs["name"], guidance, good, bad, inputs["alpha"], switch, goal, contrast, before))
        return cls(model, steps, main_camera, device, action_chunk_length)

    @staticmethod
    def _switch_embeddings(model, dataset, embedding, device, max_examples):
        """The embeddings of the last frames of the conditions, for the switch."""
        frames = condition_last_frames(dataset)[:max_examples]
        states = [state for state, _ in frames]
        actions = [actions for _, actions in frames] if embedding == "predicted" else None
        return embed_states(model, states, actions, device)

    def reset(self):
        self.current = 0
        self.step_start = 0
        self.switches = list()
        self.trace = list() # (t, step, score, d_goal, d_contrast) of every check, the score as GoalStepGuidance.check
        self.oracle_seen = None # env step at which the oracle saw the current step done
        for step in self.steps:
            if step.criterion is not None:
                step.criterion.reset()

    @property
    def step(self):
        return self.steps[self.current]

    @property
    def good_embeddings(self):
        return self.step.good_embeddings

    @property
    def bad_embeddings(self):
        return self.step.bad_embeddings

    def __call__(self, states, actions):
        return self.step.guidance(states, actions)

    def _query(self, obs, last_action):
        state = observation_state(obs, self.main_camera)
        actions = None
        if self.step.switch["embedding"] == "predicted":
            action = np.zeros(7) if last_action is None else np.asarray(last_action).reshape(-1)
            actions = [hold_chunk(action, self.action_chunk_length)]
        return embed_states(self.model, [state], actions, self.device)

    def observe(self, obs, t, last_action = None, oracle_steps_completed = None):
        """
        Looks at the observation after env step t and moves on to the next goal step if the current one is done.
        - oracle_steps_completed: how many steps the env saw done, used by the oracle switch
        Returns True if it moved on, so the caller can make the policy plan again.
        """
        if self.current >= len(self.steps) - 1: # the last step is guided until the env ends the episode
            return False
        switch = self.step.switch
        if switch["mode"] == "oracle":
            if oracle_steps_completed is not None and oracle_steps_completed > self.current:
                if self.oracle_seen is None:
                    self.oracle_seen = t
                if t - self.oracle_seen >= switch["delay"]:
                    self._move_on(t, {"reason" : "oracle", "env_done" : self.oracle_seen})
                    return True
            return False

        if t - self.step_start < switch["min_steps"] or t % switch["check_every"] != 0:
            return False
        done, score, d_goal, d_contrast = self.step.check(self._query(obs, last_action))
        self.trace.append((t, self.current, score, d_goal, d_contrast))
        if done:
            self._move_on(t, {"reason" : self.step.criterion.criterion, "score" : score, "d_goal" : d_goal, "d_contrast" : d_contrast})
            return True
        return False

    def _move_on(self, t, info):
        self.switches.append({"t" : t, "from_step" : self.current, "from" : self.step.name, **info})
        print(f"DynaGuide moves on at step {t}: {self.step.name} -> {self.steps[self.current + 1].name} ({info['reason']})")
        self.current += 1
        self.step_start = t
        self.oracle_seen = None

    def log(self):
        """What run_dynaguide.py writes about the switching of this episode."""
        return {"switches" : self.switches, "final_step" : self.current,
                "trace" : [[t, step, round(score, 3), round(d_goal, 2), None if d_contrast is None else round(d_contrast, 2)]
                           for t, step, score, d_goal, d_contrast in self.trace]}
