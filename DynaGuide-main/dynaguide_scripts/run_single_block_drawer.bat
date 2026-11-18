@echo off
rem Multi-goal task in one fixed scene with one block: open the drawer, pick up the blue block, put it in the drawer, close the drawer.
rem See "Multigoal guidance.ipynb", section 12. Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_single_block_drawer.bat
rem
rem [1/3] Block poses for the robot_poses collection: robots holding a lifted block, and robots that just let go of a block in
rem       the drawer, from the CALVIN debug set, all guidance conditions and all rollouts in results\
rem       -> datasets\robot_poses\blocks
rem [2/3] The guidance conditions of the four steps, rendered in the exact scene the experiment resets to (door left, switch off,
rem       lights off, only the blue block at the right front corner of the table), from robot_poses\ground_truth and robot_poses\blocks
rem       -> datasets\guidance_conditions\single_block_drawer
rem [3/3] The rollouts: latent switch, oracle switch and base policy, 20 each, 800 steps
rem       -> results\single_block_drawer
rem
rem Steps 1 and 2 are skipped when their folder already exists (delete it to make it again). Step 3 continues where it stopped
rem when you start the script again. Look at the .png files of steps 1 and 2 before the rollouts take the machine for hours.

setlocal
cd /d "%~dp0.."

rem the trained models, change these if they are somewhere else
set AGENT=base_policy_300.pth
set GUIDANCE=dynamics_model_5900.pth

set DEBUG=datasets\calvin_debug_dataset
set POSES=datasets\robot_poses
set CONDITIONS=datasets\guidance_conditions\single_block_drawer
set RESULTS=results\single_block_drawer

if /i not "%CONDA_DEFAULT_ENV%"=="dynaguide" echo Note: the active conda env is "%CONDA_DEFAULT_ENV%", not dynaguide.

for %%F in (%AGENT% %GUIDANCE% %POSES%\ground_truth\drawer_open.hdf5 %POSES%\ground_truth\drawer_close.hdf5) do (
    if not exist "%%F" (
        echo Missing DynaGuide-main\%%F
        echo The ground truth poses come from extract_robot_poses.py, see "Create guidance conditions.ipynb", section 7.
        exit /b 1
    )
)

echo.
if exist "%POSES%\blocks\" (
    echo [1/3] %POSES%\blocks is already there, skipped
) else (
    echo [1/3] Block poses from the debug set, the guidance conditions and the rollouts
    set SOURCES=datasets\guidance_conditions results
    if exist "%DEBUG%\validation\" set SOURCES=%DEBUG%\validation %DEBUG%\training datasets\guidance_conditions results
    call python dynaguide_scripts\extract_block_poses.py --sources %%SOURCES%% --output_folder %POSES%\blocks || goto :failed
)

echo.
if exist "%CONDITIONS%\" (
    echo [2/3] %CONDITIONS% is already there, skipped
) else (
    echo [2/3] Guidance conditions in the scene of the experiment
    python dynaguide_scripts\render_scene_conditions.py --spec calvin_exp_configs_examples\scene_conditions\single_block_drawer.json ^
        --output_folder %CONDITIONS% || goto :failed
)

echo.
echo [3/3] Rollouts, this takes hours. Ctrl+C stops, the same command continues
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\multigoal\plan_single_block_drawer.json ^
    --output_folder %RESULTS% --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo Done. The table above is also in %CD%\%RESULTS%\summary.json
exit /b 0

:failed
echo.
echo This step failed, see the message above. Delete the folder of this step if it was left half written, then start again.
exit /b 1
