@echo off
rem Long runs, part 1 of 2: button_drawer_switch and switch_button_drawer in random scenes. See calvin_exp_configs_examples\tasks\long_runs
rem for the tasks and why, and calvin_exp_configs_examples\multigoal\plan_long_runs_1.json for the plan. Part 2 runs on another
rem workstation with run_long_runs_2.bat.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_long_runs_1.bat
rem
rem [1/3] the two tasks, rendered in 4 random scenes each, skipped when they are there
rem [2/3] per task 5 x 100 guided rollouts (seeds 0 to 4) and 100 of the base policy, up to 800 steps, no video.
rem       About 25 hours -> results\long_runs_1
rem [3/3] the videos, rendered from the saved states with 8 workers, a frame every 4 steps, about 2 hours -> videos\ in each run folder
rem Ctrl+C stops, the same command continues where it stopped (the rollout task that was running starts again).

setlocal
cd /d "%~dp0.."

set AGENT=base_policy_300.pth
set GUIDANCE=dynamics_model_5900.pth
set CONDITIONS=datasets\guidance_conditions

if /i not "%CONDA_DEFAULT_ENV%"=="dynaguide" echo Note: the active conda env is "%CONDA_DEFAULT_ENV%", not dynaguide.
for %%F in (%AGENT% %GUIDANCE%) do if not exist "%%F" (
    echo Missing DynaGuide-main\%%F
    exit /b 1
)
if not exist "datasets\robot_poses\articulated\poses\switch_off.hdf5" (
    echo Missing DynaGuide-main\datasets\robot_poses. Copy it from the MacBook
    exit /b 1
)

echo.
echo [1/3] The tasks
for %%T in (button_drawer_switch switch_button_drawer) do (
    if exist "%CONDITIONS%\long_%%T\experiment.json" (
        echo     %%T is already there, skipped
    ) else (
        if exist "%CONDITIONS%\long_%%T" rmdir /s /q "%CONDITIONS%\long_%%T"
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\long_runs\%%T.json --output_folder %CONDITIONS%\long_%%T || goto :failed
    )
)

echo.
echo [2/3] Rollouts
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_long_runs_1.json ^
    --output_folder results\long_runs_1 --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo [3/3] Videos
python dynaguide_scripts\render_rollout_videos.py --results results\long_runs_1 --workers 8 || goto :failed

echo.
echo Done. Copy results\long_runs_1 to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Start the same command again to continue.
exit /b 1
