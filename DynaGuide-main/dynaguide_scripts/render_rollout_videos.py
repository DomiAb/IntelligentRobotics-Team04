import argparse
import glob
import json
import multiprocessing
import os
import time

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

"""
Renders the videos of rollouts afterwards, from the states saved in the rollout files, so the rollouts themselves can run without
video (run_dynaguide.py --no_video). Adapted from robomimic/robomimic/scripts/playback_dataset.py, which can't be used as it is: it
loads the "states" of each demo into the simulator, but run_dynaguide.py stores zeros there, and it builds the env through robomimic's
env types, which have no CALVIN env. Here each frame is put into the simulator from obs/states (scene) and obs/proprio (robot), as
recombine_conditions.Simulator does, without a physics step, so the frame shows exactly the recorded state.

The frames are those of run_dynaguide.py: the third_person camera at 640x640 after every video_skip-th env step, plus the last one when
the episode was decided before the horizon, and for multi-goal runs the label with the goal step DynaGuide guided towards and the steps
the env saw done, from the run's *_log.json. The dynamics model collage isn't drawn (the runs use --no_collage anyway).

One video per episode, so the episodes render in parallel (--workers, one simulator each) and a single episode can be looked up:
    <run folder>/videos/<timestamp>_demo_<n>.mp4

    python dynaguide_scripts/render_rollout_videos.py --results results/goal_switch_patches --workers 8
    python dynaguide_scripts/render_rollout_videos.py --results results/x/chain/patches --episodes 3 17

--results takes run folders (with *_rollout.hdf5 in them) or folders above them, they are searched. Videos that are already there are
skipped, so a stopped run continues.
"""

SIZE = 128 * 5 # as render_video_frame in run_dynaguide.py


def find_rollouts(paths):
    found = list()
    for path in paths:
        if path.endswith("_rollout.hdf5"):
            found.append(path)
        else:
            found += sorted(glob.glob(os.path.join(path, "**", "*_rollout.hdf5"), recursive = True))
    return found


def run_settings(rollout_file):
    """video_skip and horizon of the run and its log, with the goal steps and switches of each episode of a multi-goal run."""
    folder = os.path.dirname(rollout_file)
    run_args = dict()
    if os.path.isfile(os.path.join(folder, "args.json")):
        with open(os.path.join(folder, "args.json")) as f:
            run_args = json.load(f)
    log_file = rollout_file.replace("_rollout.hdf5", "_log.json")
    log = None
    if os.path.isfile(log_file):
        with open(log_file) as f:
            log = json.load(f)
    return run_args.get("video_skip") or 5, run_args.get("horizon"), log


def labels(log, episode):
    """
    label_after(t): the label run_dynaguide.py computes after env step t (1-based): the goal step guided then and the steps the env had
    seen done. None for single-goal runs. Also whether the episode was decided (all steps done or failed), None if the log doesn't say.
    """
    if log is None or "Episode_Goal_Steps" not in log or "Goal_Sequence" not in log:
        return None, None
    names = log["Goal_Sequence"]
    goal_steps = log["Episode_Goal_Steps"][episode]
    switches = log["Episode_Switches"][episode]["switches"] if "Episode_Switches" in log else []
    env_done = [step["completed_at"] for step in goal_steps["steps"] if step["completed_at"] is not None] # tracker t, after env step t
    def label_after(t): # after env step t (1-based, as observe() and the tracker count them)
        guided = sum(1 for switch in switches if switch["t"] <= t)
        done = sum(1 for completed in env_done if completed <= t)
        return f"guiding: {names[min(guided, len(names) - 1)]} | env done: {done}/{len(names)}"
    decided = goal_steps.get("success") or goal_steps.get("failure") is not None
    return label_after, decided


def draw_label(image, label):
    import cv2
    import numpy as np
    for color, thickness in [((0, 0, 0), 4), ((255, 255, 255), 1)]:
        image = cv2.putText(np.ascontiguousarray(image), label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness, cv2.LINE_AA)
    return image


_SIM = None


def worker_simulator(env_config):
    global _SIM
    if _SIM is None:
        from recombine_conditions import Simulator, make_env
        _SIM = Simulator(make_env(env_config))
    return _SIM


def render_episode(job):
    rollout_file, demo, episode, output, video_skip, horizon, env_config = job
    import cv2
    import h5py
    import imageio
    import numpy as np
    sim = worker_simulator(env_config)
    _, _, log = run_settings(rollout_file)
    with h5py.File(rollout_file, "r") as f:
        group = f["data"][demo]
        states, proprio = group["next_obs"]["states"][()], group["next_obs"]["proprio"][()] # the state after each env step
    n = len(states)
    label_after, decided = labels(log, episode)
    if decided is None: # a single-goal run ends before the horizon only when its goal is decided
        decided = horizon is not None and n < horizon
    # run_dynaguide.py renders after env step i (0-based) when i % video_skip == 0, with the label of the step before, and once more
    # after the last step of a decided episode, with the label of that step
    frames = [(i, None if i == 0 or label_after is None else label_after(i)) for i in range(0, n, video_skip)]
    if decided:
        frames.append((n - 1, None if label_after is None else label_after(n)))
    start = time.time()
    temporary = output + ".part.mp4"
    with imageio.get_writer(temporary, fps = 20) as writer:
        for i, label in frames:
            sim.set(states[i].astype(np.float64), proprio[i].astype(np.float64))
            image = sim.env.render(mode = "rgb_array", height = SIZE, width = SIZE)
            image = cv2.resize(image, (SIZE, SIZE))
            if label is not None:
                image = draw_label(image, label)
            writer.append_data(image)
    os.replace(temporary, output)
    return output, len(frames), time.time() - start


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs = "+", required = True, help = "run folders, folders above them, or *_rollout.hdf5 files")
    parser.add_argument("--episodes", type = int, nargs = "+", default = None, help = "only these episodes (demo_<n>), default all")
    parser.add_argument("--workers", type = int, default = max(1, min(8, os.cpu_count() // 2)), help = "simulators rendering in parallel")
    parser.add_argument("--video_skip", type = int, default = None, help = "default: the run's video_skip from args.json")
    parser.add_argument("--env_config", default = None, help = "default: configs/calvin.json, as the other scripts")
    args = parser.parse_args()

    import h5py
    from recombine_conditions import DEFAULT_ENV_CONFIG
    env_config = args.env_config or DEFAULT_ENV_CONFIG
    jobs = list()
    for rollout_file in find_rollouts(args.results):
        video_skip, horizon, _ = run_settings(rollout_file)
        video_skip = args.video_skip or video_skip
        folder = os.path.join(os.path.dirname(rollout_file), "videos")
        timestamp = os.path.basename(rollout_file)[:-len("_rollout.hdf5")]
        with h5py.File(rollout_file, "r") as f:
            demos = sorted(f["data"].keys(), key = lambda name: int(name.split("_")[-1]))
        for demo in demos:
            episode = int(demo.split("_")[-1])
            if args.episodes is not None and episode not in args.episodes:
                continue
            output = os.path.join(folder, f"{timestamp}_{demo}.mp4")
            if os.path.isfile(output):
                continue
            os.makedirs(folder, exist_ok = True)
            jobs.append((rollout_file, demo, demos.index(demo), output, video_skip, horizon, env_config))
    print(f"{len(jobs)} videos to render with {args.workers} workers")
    if len(jobs) == 0:
        return
    start = time.time()
    if args.workers == 1:
        results = map(render_episode, jobs)
    else:
        pool = multiprocessing.get_context("spawn").Pool(args.workers)
        results = pool.imap_unordered(render_episode, jobs)
    for done, (output, n_frames, seconds) in enumerate(results, start = 1):
        elapsed = time.time() - start
        print(f"[{done}/{len(jobs)}] {output}: {n_frames} frames in {seconds:.0f} s, about {elapsed / done * (len(jobs) - done) / 60:.0f} min left")
    if args.workers != 1:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
