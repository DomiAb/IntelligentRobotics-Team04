@echo off
rem Does guidance raise how often the policy grips the blue block? A check of DynaGuide itself, before tuning multi-goal guidance around
rem it. See "Multigoal guidance.ipynb", section 13. Run it in the Anaconda Prompt with the dynaguide env active, from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_grip_scale_sweep.bat
rem
rem [1/3] the base sets of robot_poses (setup_robot_poses.bat, skipped when they are there)
rem [2/3] the grip-only task: calvin_exp_configs_examples\tasks\blue_grip_table_right.json -> datasets\guidance_conditions\blue_grip_table_right
rem [3/3] calvin_exp_configs_examples\multigoal\plan_grip_scale_sweep.json: the base policy and guided at scale 1.5, 3 and 5, 20 rollouts
rem       of 200 steps each, all from the same start scenes -> results\grip_scale_sweep
rem
rem Reading the result (successes = rollouts in which the env counts the block as held): unguided, the base policy gripped in 2 of 10
rem rollouts on the MacBook. If the guided runs grip clearly more often, and more with a higher scale, the guidance works and only needs
rem strength. If they stay at the base rate, gripping is beyond what DINO + this dynamics model + this policy can be guided to.
rem Step 3 continues where it stopped when you start the script again.

setlocal
cd /d "%~dp0.."

rem the trained models, change these if they are somewhere else
set AGENT=base_policy_300.pth
set GUIDANCE=dynamics_model_5900.pth
set TASK=blue_grip_table_right

if /i not "%CONDA_DEFAULT_ENV%"=="dynaguide" echo Note: the active conda env is "%CONDA_DEFAULT_ENV%", not dynaguide.
for %%F in (%AGENT% %GUIDANCE%) do if not exist "%%F" (
    echo Missing DynaGuide-main\%%F
    exit /b 1
)

echo.
echo [1/3] robot_poses
call "%~dp0setup_robot_poses.bat" || goto :failed

echo.
echo [2/3] The task
if exist "datasets\guidance_conditions\%TASK%\experiment.json" (
    echo     %TASK% is already there, skipped
) else (
    python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\%TASK%.json || goto :failed
)

echo.
echo [3/3] Rollouts, about 4 x 20 rollouts. Ctrl+C stops, the same command continues
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_grip_scale_sweep.json ^
    --output_folder results\grip_scale_sweep --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo Done. Compare the successes of base_policy with scale_1_5, scale_3 and scale_5 in the table above (also in
echo %CD%\results\grip_scale_sweep\summary.json). The videos are in the folders of each run.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Delete the folder of this step if it was left half written, then start again.
exit /b 1
