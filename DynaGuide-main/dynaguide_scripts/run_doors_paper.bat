@echo off
rem The doors like in the paper: door_left and door_right, one behavior per rollout. 50 rollouts of up to 400 steps per task.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\run_doors_paper.bat
rem
rem It needs the robot_poses collection with 20 poses per set. Copy the folder datasets\robot_poses from the MacBook first
rem (made there with extend_robot_poses.py, the D split isn't needed here).
rem
rem [1/2] calvin_exp_configs_examples\experiments\plan_doors_paper.json -> results\doors_paper, for each door:
rem         original      the 20 robot poses as they were recorded, as in the paper
rem         recombined    the same 20 robot poses rendered without blocks in all 8 scenes
rem         base_policy   no guidance, from the same start scenes
rem [2/2] score_paper_rule.py: the rule of the authors, halfway, and the full slide -> results\doors_paper\paper_rule.json
rem
rem Step 1 continues where it stopped when you start the script again.

setlocal
cd /d "%~dp0.."

rem the trained models, change these if they are somewhere else
set AGENT=base_policy_300.pth
set GUIDANCE=dynamics_model_5900.pth
set POSES=datasets\robot_poses\articulated

if /i not "%CONDA_DEFAULT_ENV%"=="dynaguide" echo Note: the active conda env is "%CONDA_DEFAULT_ENV%", not dynaguide.
for %%F in (%AGENT% %GUIDANCE%) do if not exist "%%F" (
    echo Missing DynaGuide-main\%%F
    exit /b 1
)
for %%D in (door_left door_right) do if not exist "%POSES%\poses\%%D.hdf5" (
    echo Missing DynaGuide-main\%POSES%\poses\%%D.hdf5
    echo Copy the new folder datasets\robot_poses from the MacBook. The old one has only 3 to 4 door poses.
    exit /b 1
)

echo.
echo [1/2] Rollouts, 6 tasks x 50 rollouts. Ctrl+C stops, the same command continues
python dynaguide_scripts\run_experiments.py --plan calvin_exp_configs_examples\experiments\plan_doors_paper.json ^
    --output_folder results\doors_paper --agent %AGENT% --guidance %GUIDANCE% || goto :failed

echo.
echo [2/2] Scores
python dynaguide_scripts\score_paper_rule.py --results results\doors_paper || goto :failed

echo.
echo Done. Copy results\doors_paper to the MacBook.
exit /b 0

:failed
echo.
echo This step failed, see the message above. Start the same command again to continue.
exit /b 1
