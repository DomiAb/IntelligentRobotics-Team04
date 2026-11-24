@echo off
rem Open the drawer, grip the blue block, let go of it in the drawer, close the drawer, with only the blue block. See "Multigoal guidance.ipynb", sections 1 and 12.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_single_block_drawer.bat
rem
rem [1/3] the base sets of robot_poses (setup_robot_poses.bat)
rem [2/3] the task: guidance conditions in the scene of the reset, following the story of the steps, and the experiment config
rem       calvin_exp_configs_examples\tasks\single_block_drawer.json -> datasets\guidance_conditions\single_block_drawer
rem [3/3] the rollouts of calvin_exp_configs_examples\multigoal\plan_single_block_drawer.json: latent switch, oracle switch, base policy, 20 each -> results\single_block_drawer
rem
rem Steps 1 and 2 are skipped when their output already exists (delete the folder to make it again). Step 3 continues where it
rem stopped when you start the script again. Look at <task>_all.png of step 2 before the rollouts take the machine for hours.

setlocal
cd /d "%~dp0.."

rem the trained models, change these if they are somewhere else
set AGENT=base_policy_300.pth
set GUIDANCE=dynamics_model_5900.pth

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
if exist "datasets\guidance_conditions\single_block_drawer\experiment.json" (
    echo     single_block_drawer is already there, skipped
) else (
    python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\single_block_drawer.json || goto :failed
)

echo.
echo [3/3] Rollouts, this takes hours. Ctrl+C stops, the same command continues
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_single_block_drawer.json ^
    --output_folder results\single_block_drawer --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo Done. The table above is also in %CD%\results\single_block_drawer\summary.json
exit /b 0

:failed
echo.
echo This step failed, see the message above. Delete the folder of this step if it was left half written, then start again.
exit /b 1
