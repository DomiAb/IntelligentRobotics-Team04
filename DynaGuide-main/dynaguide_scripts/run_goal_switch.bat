@echo off
rem Tests the goal switch (the change criterion): does DynaGuide notice by itself that a goal is reached? See "Multigoal guidance.ipynb",
rem section 7 for how it works and section 8 for the tests.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_goal_switch.bat
rem
rem Four chains of three steps, no blocks. Each of the eight joint behaviors is once a step that the switch has to end:
rem     drawer_button_door      open the drawer, button on,        door right
rem     door_drawer_switch      door left,       close the drawer, switch on
rem     button_switch_drawer    button off,      switch off,       open the drawer
rem     door_switch_button      door right,      switch on,        button off
rem
rem [1/6] the base sets of robot_poses (setup_robot_poses.bat, skipped when they are there)
rem [2/6] the four tasks with their before_examples -> datasets\guidance_conditions\<chain>
rem       A drawer_button_door from before this switch is renamed to drawer_button_door_without_before, not deleted
rem [3/6] offline, no rollouts: the switch over the recordings of the debug set and of the D split, if they are on this machine
rem       -> results\goal_switch_recordings
rem [4/6] rollouts with the new switch and with the oracle, 8 x 20 of up to 800 steps, about 10 hours -> results\goal_switch
rem       then what the switch did in them (switch_live.json)
rem [5/6] the same with the switch from before, 4 x 20, about 5 hours -> results\goal_switch_previous
rem [6/6] what the switch did in all of them, and the new switch over every frame of every rollout, about 1.5 hours
rem       -> results\goal_switch\switch_live.json and switch_replay.json
rem
rem Everything continues where it stopped when you start the script again. "run_goal_switch.bat short" leaves out step 5, then step 6
rem only looks at the rollouts of step 4.

setlocal
cd /d "%~dp0.."

rem the trained models, change these if they are somewhere else
set AGENT=base_policy_300.pth
set GUIDANCE=dynamics_model_5900.pth
set PLANS=calvin_exp_configs_examples\multigoal
set RESULTS=results\goal_switch
if /i not "%~1"=="short" set RESULTS=results\goal_switch results\goal_switch_previous

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
echo [1/6] robot_poses
call "%~dp0setup_robot_poses.bat" || goto :failed

echo.
echo [2/6] The tasks
if exist "datasets\guidance_conditions\drawer_button_door\experiment.json" if not exist "datasets\guidance_conditions\drawer_button_door\open_drawer_before.hdf5" (
    echo     drawer_button_door is from before this switch, renamed
    ren "datasets\guidance_conditions\drawer_button_door" drawer_button_door_without_before || goto :failed
)
for %%T in (drawer_button_door door_drawer_switch button_switch_drawer door_switch_button) do (
    if exist "datasets\guidance_conditions\%%T\experiment.json" (
        echo     %%T is already there, skipped
    ) else (
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\%%T.json || goto :failed
    )
)

echo.
echo [3/6] Offline: the switch over recordings
set RECORDINGS=
if exist "datasets\calvin_debug_dataset\validation\ep_start_end_ids.npy" set RECORDINGS=datasets\calvin_debug_dataset\validation
if exist "datasets\D_Split\task_D_D\validation\ep_start_end_ids.npy" set RECORDINGS=%RECORDINGS% datasets\D_Split\task_D_D\validation
if "%RECORDINGS%"=="" (
    echo     neither the debug set nor the D split is on this machine, skipped
) else if exist "results\goal_switch_recordings\switch_replay.json" (
    echo     already there, skipped
) else (
    python dynaguide_scripts\test_goal_switch.py recordings --recordings %RECORDINGS% --guidance %GUIDANCE% --stride 3 --output results\goal_switch_recordings || echo     This part failed, the rollouts don't need it, going on
)

echo.
echo [4/6] Rollouts with the new switch and the oracle. Ctrl+C stops, the same command continues
python dynaguide_scripts\run_experiments.py --plan %PLANS%\plan_goal_switch.json --output_folder results\goal_switch ^
    --agent %AGENT% --guidance %GUIDANCE% || goto :failed
python dynaguide_scripts\test_goal_switch.py live --results results\goal_switch || goto :failed

echo.
echo [5/6] Rollouts with the switch from before
if /i "%~1"=="short" (
    echo     left out
) else (
    python dynaguide_scripts\run_experiments.py --plan %PLANS%\plan_goal_switch_previous.json --output_folder results\goal_switch_previous ^
        --agent %AGENT% --guidance %GUIDANCE% || goto :failed
)

echo.
echo [6/6] What the switch did, and the new switch over every rollout
python dynaguide_scripts\test_goal_switch.py live --results %RESULTS% || goto :failed
python dynaguide_scripts\test_goal_switch.py replay --results %RESULTS% --guidance %GUIDANCE% || goto :failed

echo.
echo Done. Copy results\goal_switch*, all folders that start like this, to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Start the same command again to continue.
exit /b 1
