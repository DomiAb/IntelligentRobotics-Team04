@echo off
rem Tests the patches criterion of the goal switch: the same question as run_goal_switch.bat, but the score only looks at the patches
rem the step changes, and the best fitting pair of images counts. See "Multigoal guidance.ipynb", section 7.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_goal_switch_patches.bat
rem
rem Use another computer, or another copy of the project, while run_goal_switch.bat is still running: it starts Python again for every
rem part and would pick up this code halfway. The rollouts here don't need its results.
rem
rem [1/5] the base sets of robot_poses (setup_robot_poses.bat, skipped when they are there)
rem [2/5] the four tasks of run_goal_switch.bat with their before_examples, skipped when they are there
rem [3/5] 4 x 20 rollouts of up to 800 steps with the patches criterion, about 5 hours -> results\goal_switch_patches
rem       then what the switch did in them (switch_live.json)
rem [4/5] both criteria over every frame of these rollouts, and of results\goal_switch if that folder is there. About 0.5 to 1.5 hours
rem       -> results\goal_switch_patches\switch_replay.json (change) and switch_replay_patches.json (patches)
rem [5/5] releases, no rollouts: the criteria over the rollouts of results\blue_drawer_end and results\blue_drawer_end_1.1, if they are
rem       there (the robot rarely gets the block into the drawer). About 1 hour -> results\goal_switch_patches_release
rem
rem Steps 2 and 3 continue where they stopped when you start the script again. Step 4 always runs, step 5 is skipped once it is done.

setlocal
cd /d "%~dp0.."

rem the trained models, change these if they are somewhere else
set AGENT=base_policy_300.pth
set GUIDANCE=dynamics_model_5900.pth
set CONDITIONS=datasets\guidance_conditions

if /i not "%CONDA_DEFAULT_ENV%"=="dynaguide" echo Note: the active conda env is "%CONDA_DEFAULT_ENV%", not dynaguide.
for %%F in (%AGENT% %GUIDANCE%) do if not exist "%%F" (
    echo Missing DynaGuide-main\%%F
    exit /b 1
)
if not exist "datasets\robot_poses\articulated\poses\switch_off.hdf5" (
    echo Missing DynaGuide-main\datasets\robot_poses\articulated\poses. Copy the new folder datasets\robot_poses from the MacBook.
    exit /b 1
)

echo.
echo [1/5] robot_poses
call "%~dp0setup_robot_poses.bat" || goto :failed

echo.
echo [2/5] The tasks
if exist "%CONDITIONS%\drawer_button_door\experiment.json" if not exist "%CONDITIONS%\drawer_button_door\open_drawer_before.hdf5" (
    echo     drawer_button_door has no before images, renamed
    ren "%CONDITIONS%\drawer_button_door" drawer_button_door_without_before || goto :failed
)
for %%T in (drawer_button_door door_drawer_switch button_switch_drawer door_switch_button) do (
    if exist "%CONDITIONS%\%%T\experiment.json" (
        echo     %%T is already there, skipped
    ) else (
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\%%T.json || goto :failed
    )
)

echo.
echo [3/5] Rollouts with the patches criterion. Ctrl+C stops, the same command continues
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_goal_switch_patches.json ^
    --output_folder results\goal_switch_patches --agent %AGENT% --guidance %GUIDANCE% || goto :failed
python dynaguide_scripts\test_goal_switch.py live --results results\goal_switch_patches || goto :failed

echo.
echo [4/5] Both criteria over every frame of the rollouts
set RESULTS=results\goal_switch_patches
if exist "results\goal_switch\summary.json" set RESULTS=results\goal_switch_patches results\goal_switch
python dynaguide_scripts\test_goal_switch.py replay --results %RESULTS% --guidance %GUIDANCE% --output results\goal_switch_patches || goto :failed

echo.
echo [5/5] Releases in recorded rollouts
if not exist "results\blue_drawer_end\summary.json" goto :no_release
if not exist "results\blue_drawer_end_1.1\summary.json" goto :no_release
if exist "results\goal_switch_patches_release\center\switch_replay_patches.json" (
    echo     already there, skipped
    goto :finished
)
for %%T in (right center) do (
    if not exist "%CONDITIONS%\blue_drawer_end_%%T_patches\experiment.json" (
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\blue_drawer_end_%%T.json --output_folder %CONDITIONS%\blue_drawer_end_%%T_patches || goto :failed
    )
    python dynaguide_scripts\test_goal_switch.py replay --results results\blue_drawer_end results\blue_drawer_end_1.1 --experiments %%T ^
        --task %CONDITIONS%\blue_drawer_end_%%T_patches --guidance %GUIDANCE% --stride 5 --output results\goal_switch_patches_release\%%T || goto :failed
)
goto :finished

:no_release
echo     results\blue_drawer_end and results\blue_drawer_end_1.1 aren't on this machine, skipped. Copy them from the MacBook to run this part
goto :finished

:finished
echo.
echo Done. Copy results\goal_switch_patches*, all folders that start like this, to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Start the same command again to continue.
exit /b 1
