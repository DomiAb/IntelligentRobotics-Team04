@echo off
rem Random scenes: how DynaGuide does when the parts of the scene a task doesn't use are random, and the guidance conditions show such
rem random scenes too. Random means: the joints the task doesn't need at the start at a random end, all three blocks somewhere on the table.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_random_scenes.bat
rem
rem [1/4] the tasks of calvin_exp_configs_examples\tasks\random_scenes, skipped when they are there. The random ones are rendered in 4
rem       random scenes each (make_task.py, random_scenes)
rem [2/4] single goals, drawer_open, button_off and switch_on, 20 rollouts of up to 400 steps each in three ways, no video:
rem       fixed (the fixed scene and its conditions), random_env (random scene, fixed conditions), random (random scene and conditions)
rem       About 2 hours -> results\random_scenes_singles
rem [3/4] the chain button_switch_drawer in random scenes with random conditions, 30 rollouts of up to 800 steps, no video, about
rem       1.5 hours -> results\random_scenes_chain
rem [4/4] the videos, rendered from the saved states with 8 workers, about 20 minutes -> videos\ in each run folder
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
for %%T in (fixed_drawer_open fixed_button_off fixed_switch_on random_drawer_open random_button_off random_switch_on random_button_switch_drawer) do (
    if exist "%CONDITIONS%\%%T\experiment.json" (
        echo     %%T is already there, skipped
    ) else (
        if exist "%CONDITIONS%\%%T" rmdir /s /q "%CONDITIONS%\%%T"
        python dynaguide_scripts\make_task.py --task calvin_exp_configs_examples\tasks\random_scenes\%%T.json || goto :failed
    )
)

echo.
echo [2/4] Single goals
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_random_singles.json ^
    --output_folder results\random_scenes_singles --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo [3/4] The chain
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_random_chain.json ^
    --output_folder results\random_scenes_chain --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo [4/4] Videos
python dynaguide_scripts\render_rollout_videos.py --results results\random_scenes_singles results\random_scenes_chain --workers 8 || goto :failed

echo.
echo Done. Copy results\random_scenes_singles and results\random_scenes_chain to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Start the same command again to continue.
exit /b 1
