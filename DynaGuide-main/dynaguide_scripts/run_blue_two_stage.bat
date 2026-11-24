@echo off
rem Two-stage grip at scale 5: grip the blue block on the table, let go of it in the open drawer. Follows the grip scale sweep,
rem where only scale 5 gripped (6/20, base policy 0/20). See "Multigoal guidance.ipynb", section 13.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_blue_two_stage.bat
rem
rem [1/3] the base sets of robot_poses (setup_robot_poses.bat, skipped when they are there)
rem [2/3] the tasks, without a scale of their own so the plan's scale 5 applies to every step
rem       calvin_exp_configs_examples\tasks\blue_two_stage.json -> datasets\guidance_conditions\blue_two_stage
rem       calvin_exp_configs_examples\tasks\blue_two_stage_one_step.json -> datasets\guidance_conditions\blue_two_stage_one_step
rem [3/3] calvin_exp_configs_examples\multigoal\plan_blue_two_stage.json: two steps with the latent and the oracle switch, one step,
rem       the base policy, 20 rollouts of 500 steps each, all from the same start scenes -> results\blue_two_stage
rem
rem Steps 1 and 2 are skipped when their output already exists (delete the folder to make it again). Step 3 continues where it
rem stopped when you start the script again.

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
echo [2/3] The tasks
for %%T in (blue_two_stage blue_two_stage_one_step) do (
    if exist "datasets\guidance_conditions\%%T\experiment.json" (
        echo     %%T is already there, skipped
    ) else (
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\%%T.json || goto :failed
    )
)

echo.
echo [3/3] Rollouts, 4 x 20 rollouts of up to 500 steps, a few hours. Ctrl+C stops, the same command continues
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_blue_two_stage.json ^
    --output_folder results\blue_two_stage --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo Done. The table above is also in %CD%\results\blue_two_stage\summary.json. Copy results\blue_two_stage to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Delete the folder of this step if it was left half written, then start again.
exit /b 1
