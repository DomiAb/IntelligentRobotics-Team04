@echo off
rem Sets up the base sets of the robot_poses collection, see "Multigoal guidance.ipynb", section 13. Run it in the Anaconda Prompt with
rem the dynaguide env active, from any folder. The run_*.bat scripts of the block experiments call it first.
rem
rem     datasets\robot_poses\articulated\<behavior>\<scene>.hdf5     drawer, door, switch and button at their goal, without blocks, in every
rem                                                                 combination of the other joints (recombine_conditions.py --hide_blocks)
rem     datasets\robot_poses\blocks\<kind>\<action>_<side>\<scene>.hdf5   gripping and letting go of a block on the table (left, center,
rem                                                                 right), in the drawer and in the cabinet (left, right), with the blue
rem                                                                 block only, in every combination of the joints (extract_block_poses.py)
rem
rem A set that is already there is left as it is. An old block folder (block_lifted.hdf5, or poses without base sets) goes to _old.

setlocal
cd /d "%~dp0.."
set POSES=datasets\robot_poses
set DEBUG=datasets\calvin_debug_dataset

if exist "%POSES%\articulated\drawer_open\" (
    echo %POSES%\articulated is already there
) else (
    if exist "%POSES%\articulated\" (
        echo %POSES%\articulated has no drawer_open folder. Move it away, then start again to make the sets without blocks.
        exit /b 1
    )
    echo Articulated poses without blocks, in every scene
    python dynaguide_scripts\recombine_conditions.py --hide_blocks --output_folder %POSES%\articulated || exit /b 1
)

if not exist "%POSES%\_old\" mkdir "%POSES%\_old"
if exist "%POSES%\blocks\block_lifted.hdf5" move "%POSES%\blocks" "%POSES%\_old\blocks_v1" > nul
if exist "%POSES%\blocks\grip_table.hdf5" move "%POSES%\blocks" "%POSES%\_old\blocks_v2" > nul
if exist "%POSES%\blocks\grip_table\" (
    echo %POSES%\blocks is already there
    exit /b 0
)
if exist "%POSES%\blocks\" (
    echo %POSES%\blocks exists but has no grip_table set, probably a run that stopped. Delete the folder and start again.
    exit /b 1
)
echo Block poses from the debug set, the guidance conditions and the rollouts, and their base sets
if exist "%DEBUG%\validation\" (
    python dynaguide_scripts\extract_block_poses.py --sources %DEBUG%\validation %DEBUG%\training datasets\guidance_conditions results --output_folder %POSES%\blocks || exit /b 1
) else (
    echo Note: no CALVIN debug set in %DEBUG%, the poses come from the conditions and rollouts only
    python dynaguide_scripts\extract_block_poses.py --sources datasets\guidance_conditions results --output_folder %POSES%\blocks || exit /b 1
)
echo Look at the *_all.png files in %CD%\%POSES%\blocks
exit /b 0
