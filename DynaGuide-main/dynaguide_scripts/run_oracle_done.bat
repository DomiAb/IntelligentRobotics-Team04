@echo off
rem The oracle switch with the looser success thresholds of 2026-11-26: a joint counts as done at about 87% of the way (DONE in
rem core\calvin_behaviors.py), and the oracle keeps guiding to the old step for 10 more env steps, so a joint still moving gets to its end.
rem The same four chains and start scenes as run_goal_switch.bat and run_goal_switch_patches.bat. Their rollouts don't need running again:
rem the success of the change and patches runs is scored again on the MacBook from the recorded states.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_oracle_done.bat
rem
rem [1/2] the four tasks, skipped when they are there (they are, after run_goal_switch_patches.bat)
rem [2/2] 4 x 20 rollouts of up to 800 steps, about 4 hours -> results\oracle_done
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
echo [1/2] The tasks
for %%T in (drawer_button_door door_drawer_switch button_switch_drawer door_switch_button) do (
    if exist "%CONDITIONS%\%%T\experiment.json" (
        echo     %%T is already there, skipped
    ) else (
        if not exist "datasets\robot_poses\articulated\poses\switch_off.hdf5" (
            echo Missing DynaGuide-main\datasets\robot_poses. Copy it from the MacBook, or run this on the workstation of run_goal_switch_patches.bat
            exit /b 1
        )
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\%%T.json || goto :failed
    )
)

echo.
echo [2/2] Rollouts with the oracle switch
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_oracle_done.json ^
    --output_folder results\oracle_done --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo Done. Copy results\oracle_done to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Start the same command again to continue.
exit /b 1
