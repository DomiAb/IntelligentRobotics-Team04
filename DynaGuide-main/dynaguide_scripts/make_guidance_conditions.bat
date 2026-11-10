@echo off
rem Makes all guidance conditions from the CALVIN debug set, see "Create guidance conditions.ipynb", sections 1 and 6.
rem Run it in the Anaconda Prompt with the dynaguide env active. It works from any folder:
rem     conda activate dynaguide
rem     DynaGuide-main\dynaguide_scripts\make_guidance_conditions.bat
rem
rem Reads   DynaGuide-main\datasets\calvin_debug_dataset\validation and \training
rem Writes  DynaGuide-main\datasets\guidance_conditions\
rem             debug_all                    one condition file per behavior, cut like the authors did
rem             button_pressed               the 6 button presses, as button_on and button_off
rem             recombined_debug             the robots of the drawer, door, switch and button conditions in every scene
rem             recombined_debug_no_blocks   the same without blocks
rem A folder that already exists is never overwritten: the script stops and names it. Delete or rename it to make it again.

setlocal
cd /d "%~dp0.."

set DEBUG=datasets\calvin_debug_dataset
set OUT=datasets\guidance_conditions

if /i not "%CONDA_DEFAULT_ENV%"=="dynaguide" echo Note: the active conda env is "%CONDA_DEFAULT_ENV%", not dynaguide.

for %%S in (validation training) do (
    if not exist "%DEBUG%\%%S\" (
        echo Missing DynaGuide-main\%DEBUG%\%%S. Unzip the CALVIN debug set into DynaGuide-main\datasets first.
        exit /b 1
    )
)

set FOUND=0
for %%F in (debug_all button_pressed recombined_debug recombined_debug_no_blocks) do (
    if exist "%OUT%\%%F\" (
        echo Already there: DynaGuide-main\%OUT%\%%F
        set FOUND=1
    )
)
if "%FOUND%"=="1" (
    echo Delete or rename these folders to make them again.
    exit /b 1
)

set SEEDS=%OUT%\debug_all\drawer_open.hdf5 %OUT%\debug_all\drawer_close.hdf5 %OUT%\debug_all\door_left.hdf5 %OUT%\debug_all\door_right.hdf5 %OUT%\debug_all\switch_on.hdf5 %OUT%\debug_all\switch_off.hdf5 %OUT%\button_pressed\button_on.hdf5 %OUT%\button_pressed\button_off.hdf5

echo.
echo [1/4] One condition file per behavior from the debug set
python dynaguide_scripts\create_guidance_conditions.py --recordings %DEBUG%\validation %DEBUG%\training --output_folder %OUT%\debug_all || goto :failed

echo.
echo [2/4] The button presses, as button_on and button_off
python dynaguide_scripts\create_guidance_conditions.py --recordings %DEBUG%\validation %DEBUG%\training --goals calvin_exp_configs_examples\guidance_goals\button_pressed.json --output_folder %OUT%\button_pressed || goto :failed

echo.
echo [3/4] Every scene, with blocks
python dynaguide_scripts\recombine_conditions.py --conditions %SEEDS% --output_folder %OUT%\recombined_debug || goto :failed

echo.
echo [4/4] Every scene, without blocks
python dynaguide_scripts\recombine_conditions.py --conditions %SEEDS% --hide_blocks --output_folder %OUT%\recombined_debug_no_blocks || goto :failed

echo.
echo Done. Look at the overview.png of each behavior in %CD%\%OUT%\recombined_debug and %CD%\%OUT%\recombined_debug_no_blocks
exit /b 0

:failed
echo.
echo This step failed, see the message above. The folders of the steps before it are complete, delete the folder of this step before you start again.
exit /b 1
