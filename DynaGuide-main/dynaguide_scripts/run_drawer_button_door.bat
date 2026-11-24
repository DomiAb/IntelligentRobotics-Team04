@echo off
rem Three articulated behaviors in a row, no blocks: open the drawer, press the button (green LED on), slide the door from left to
rem right. See "Multigoal guidance.ipynb", section 1.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_drawer_button_door.bat
rem
rem [1/3] the base sets of robot_poses (setup_robot_poses.bat, skipped when they are there)
rem [2/3] the task: guidance conditions in the scene of the reset, following the story of the steps, and the experiment config
rem       calvin_exp_configs_examples\tasks\drawer_button_door.json -> datasets\guidance_conditions\drawer_button_door
rem       The one-step control calvin_exp_configs_examples\multigoal\drawer_button_door_one_step.json uses its last step's conditions
rem [3/3] calvin_exp_configs_examples\multigoal\plan_drawer_button_door.json: three steps with the latent and the oracle switch,
rem       the base policy, and one step to the end state, 20 rollouts of up to 800 steps each -> results\drawer_button_door
rem
rem Steps 1 and 2 are skipped when their output already exists (delete the folder to make it again). Step 3 continues where it
rem stopped when you start the script again. Look at drawer_button_door_all.png of step 2 before the rollouts take the machine for hours.

setlocal
cd /d "%~dp0.."

rem the trained models, change these if they are somewhere else
set AGENT=base_policy_300.pth
set GUIDANCE=dynamics_model_5900.pth
set TASK=drawer_button_door

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
echo [3/3] Rollouts, 4 x 20 rollouts of up to 800 steps, several hours. Ctrl+C stops, the same command continues
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_%TASK%.json ^
    --output_folder results\%TASK% --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo Done. The table above is also in %CD%\results\%TASK%\summary.json. Copy results\%TASK% to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Delete the folder of this step if it was left half written, then start again.
exit /b 1
