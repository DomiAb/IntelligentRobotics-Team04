@echo off
rem Guide only to the end state: the blue block lying in the open drawer, however the policy gets it there (grip and place, or push
rem it off the table edge into the drawer). See "Multigoal guidance.ipynb", section 13.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_blue_drawer_end.bat
rem
rem [1/3] the base sets of robot_poses (setup_robot_poses.bat, skipped when they are there)
rem [2/3] the tasks, one step each, without a scale of their own so the plan sets it
rem       calvin_exp_configs_examples\tasks\blue_drawer_end_right.json  -> datasets\guidance_conditions\blue_drawer_end_right
rem           the scene of the grip sweep, blue on the right third of the table
rem       calvin_exp_configs_examples\tasks\blue_drawer_end_center.json -> datasets\guidance_conditions\blue_drawer_end_center
rem           blue on the center third, right in front of the drawer, where CALVIN's pushes into the drawer start
rem [3/3] calvin_exp_configs_examples\multigoal\plan_blue_drawer_end.json: right at scale 1.5, 5 and 10 plus the base policy,
rem       center at scale 5 and 10 plus the base policy, 20 rollouts of up to 300 steps each -> results\blue_drawer_end
rem
rem Success = the blue block rests on the drawer floor, held before or not. Steps 1 and 2 are skipped when their output already
rem exists (delete the folder to make it again). Step 3 continues where it stopped when you start the script again.

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
for %%T in (blue_drawer_end_right blue_drawer_end_center) do (
    if exist "datasets\guidance_conditions\%%T\experiment.json" (
        echo     %%T is already there, skipped
    ) else (
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\%%T.json || goto :failed
    )
)

echo.
echo [3/3] Rollouts, 7 x 20 rollouts of up to 300 steps, about 3 to 4 hours. Ctrl+C stops, the same command continues
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_blue_drawer_end.json ^
    --output_folder results\blue_drawer_end --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo Done. The table above is also in %CD%\results\blue_drawer_end\summary.json. Copy results\blue_drawer_end to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Delete the folder of this step if it was left half written, then start again.
exit /b 1
