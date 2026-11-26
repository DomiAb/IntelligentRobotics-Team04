@echo off
rem The spawn test: does a step later in a chain fail because of where the arm is when the step starts?
rem button_on (step 2) and door_right (step 3) of drawer_button_door, alone, from the exact scenes where DynaGuide switched to them in
rem the recorded chain runs. Each scene twice: the arm where the chain left it (chain_start), and the arm at a spawn pose (spawn).
rem If spawn clearly wins, going back to a spawn pose between the steps should help. Made with dynaguide_scripts\make_spawn_test.py.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_spawn_test.bat
rem
rem [1/3] the drawer_button_door task (its conditions), skipped when it is there
rem [2/3] 2 x 60 rollouts of button_on (up to 250 steps) and 2 x 50 of door_right (up to 400 steps), no video, about 3 hours
rem       -> results\spawn_test\button_on and results\spawn_test\door_right
rem [3/3] the videos, rendered from the saved states with 8 workers, about 15 minutes -> videos\ in each run folder
rem Ctrl+C stops, the same command continues where it stopped.

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

echo.
echo [1/3] The task
if exist "%CONDITIONS%\drawer_button_door\experiment.json" (
    echo     drawer_button_door is already there, skipped
) else (
    if not exist "datasets\robot_poses\articulated\poses\switch_off.hdf5" (
        echo Missing DynaGuide-main\datasets\robot_poses. Copy it from the MacBook
        exit /b 1
    )
    python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\drawer_button_door.json || goto :failed
)

echo.
echo [2/3] Rollouts
for %%S in (button_on door_right) do (
    python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\spawn_test\plan_%%S.json ^
        --output_folder results\spawn_test\%%S --agent %AGENT% --guidance %GUIDANCE% || goto :failed
)

echo.
echo [3/3] Videos
python dynaguide_scripts\render_rollout_videos.py --results results\spawn_test --workers 8 || goto :failed

echo.
echo Done. Copy results\spawn_test to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Start the same command again to continue.
exit /b 1
