@echo off
rem Two tests next to the long runs:
rem   homing: does guiding the arm back towards spawn poses before the button help? Open the drawer, press the button (the fixed scene of
rem           drawer_button_door), without a home phase and with 40 or 80 env steps of it. 3 x 60 rollouts of up to 600 steps
rem   door conditions: why does door_right fail after the button step of drawer_button_door? From the 50 scenes of the spawn test, the
rem           drawer open or closed in the scene, times conditions rendered with the drawer open or closed. 4 x 50 rollouts of up to 400 steps
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_homing_door_tests.bat
rem
rem [1/4] the tasks: drawer_button_door, door_switch_button and the three homing tasks, skipped when they are there
rem [2/4] the homing test, no video, about 4 hours -> results\homing_test
rem [3/4] the door conditions test, no video, about 4 hours -> results\door_conditions_test
rem [4/4] the videos, rendered from the saved states with 8 workers, a frame every 4 steps, about 30 minutes
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
if not exist "datasets\robot_poses\articulated\poses\switch_off.hdf5" (
    echo Missing DynaGuide-main\datasets\robot_poses. Copy it from the MacBook
    exit /b 1
)

echo.
echo [1/4] The tasks
for %%T in (drawer_button_door door_switch_button) do (
    if exist "%CONDITIONS%\%%T\experiment.json" (
        echo     %%T is already there, skipped
    ) else (
        if exist "%CONDITIONS%\%%T" rmdir /s /q "%CONDITIONS%\%%T"
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\%%T.json || goto :failed
    )
)
for %%T in (drawer_button drawer_home40_button drawer_home80_button) do (
    if exist "%CONDITIONS%\homing_%%T\experiment.json" (
        echo     homing_%%T is already there, skipped
    ) else (
        if exist "%CONDITIONS%\homing_%%T" rmdir /s /q "%CONDITIONS%\homing_%%T"
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\homing\%%T.json --output_folder %CONDITIONS%\homing_%%T || goto :failed
    )
)

echo.
echo [2/4] The homing test
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_homing.json ^
    --output_folder results\homing_test --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo [3/4] The door conditions test
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_door_conditions.json ^
    --output_folder results\door_conditions_test --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo [4/4] Videos
python dynaguide_scripts\render_rollout_videos.py --results results\homing_test results\door_conditions_test --workers 8 || goto :failed

echo.
echo Done. Copy results\homing_test and results\door_conditions_test to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Start the same command again to continue.
exit /b 1
