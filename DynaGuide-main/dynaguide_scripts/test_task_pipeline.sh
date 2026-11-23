#!/bin/bash
# Runs the whole pipeline of a task once, from empty folders, with few short rollouts: the commands of setup_robot_poses.bat and
# run_blue_table_to_open_drawer.bat, on macOS / Linux. To check that every step works before the long run on the workstation.
#
#     bash dynaguide_scripts/test_task_pipeline.sh                      (from DynaGuide-main, the conda env active)
#     ROLLOUTS=2 HORIZON=200 bash dynaguide_scripts/test_task_pipeline.sh
#
# Everything goes to results/pipeline_test (delete it to start again):
#     robot_poses/articulated, robot_poses/blocks   the base sets (recombine_conditions.py --hide_blocks, extract_block_poses.py)
#     tasks/<task>/                                   the conditions and experiment.json of each task (make_task.py)
#     plan.json, rollouts/                            the plan with the test's configs, few rollouts, and their results (run_experiments.py)

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/robomimic:$PWD/calvin/calvin_env:$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT=${OUT:-results/pipeline_test}
ROLLOUTS=${ROLLOUTS:-2}
HORIZON=${HORIZON:-200}
AGENT=${AGENT:-base_policy_300.pth}
GUIDANCE=${GUIDANCE:-dynamics_model_5900.pth}
PLAN=calvin_exp_configs_examples/multigoal/plan_blue_table_to_open_drawer.json
TASKS="blue_table_to_open_drawer blue_table_to_open_drawer_one_step"
DEBUG=datasets/calvin_debug_dataset

if [ -e "$OUT" ]; then echo "$OUT already exists, delete it to test from empty folders"; exit 1; fi
for f in "$AGENT" "$GUIDANCE" "$DEBUG/validation" datasets/guidance_conditions/debug_all datasets/guidance_conditions/button_pressed; do
    if [ ! -e "$f" ]; then echo "Missing $f"; exit 1; fi
done
step() { echo; echo "[$(date +%H:%M:%S)] $*"; }

step "1/5 articulated base sets, without blocks"
python dynaguide_scripts/recombine_conditions.py --hide_blocks --output_folder "$OUT/robot_poses/articulated"

step "2/5 block poses and their base sets"
python dynaguide_scripts/extract_block_poses.py --sources "$DEBUG/validation" "$DEBUG/training" datasets/guidance_conditions results \
    --output_folder "$OUT/robot_poses/blocks"

step "3/5 the tasks"
for task in $TASKS; do
    python dynaguide_scripts/make_task.py --task "calvin_exp_configs_examples/tasks/$task.json" --poses_root "$OUT/robot_poses" \
        --output_folder "$OUT/tasks/$task"
done

step "4/5 the plan with the test's configs, $ROLLOUTS rollouts of $HORIZON steps"
python - "$PLAN" "$OUT" "$ROLLOUTS" "$HORIZON" <<'EOF'
import json, sys
plan_path, out, rollouts, horizon = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(plan_path) as f:
    plan = json.load(f)
plan.update(n_rollouts = rollouts, horizon = horizon)
for entry in plan["experiments"].values():
    entry["config"] = entry["config"].replace("datasets/guidance_conditions", f"{out}/tasks")
with open(f"{out}/plan.json", "w") as f:
    json.dump(plan, f, indent = 4)
EOF
python dynaguide_scripts/run_experiments.py --plan "$OUT/plan.json" --output_folder "$OUT/rollouts" --dry_run

step "5/5 the rollouts"
python dynaguide_scripts/run_experiments.py --plan "$OUT/plan.json" --output_folder "$OUT/rollouts" --agent "$AGENT" --guidance "$GUIDANCE"

step "Done. Look at $OUT/tasks/*/*_all.png, the videos in $OUT/rollouts and $OUT/rollouts/summary.json"
