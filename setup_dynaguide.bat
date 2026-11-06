@echo off
rem One-command install of the DynaGuide env on Windows (Linux and macOS: setup_dynaguide.sh).
rem Usage: run setup_dynaguide.bat from an Anaconda Prompt
setlocal

set "ENV_NAME=dynaguide_env"
cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 goto no_conda

call conda env list | findstr /b /c:"%ENV_NAME% " >nul
if errorlevel 1 goto create

echo Updating existing env %ENV_NAME%...
call conda env update -n %ENV_NAME% -f dynaguide_env.yml
if errorlevel 1 goto failed
goto check

:create
echo Creating env %ENV_NAME%...
call conda env create -n %ENV_NAME% -f dynaguide_env.yml
if errorlevel 1 goto failed

:check
if not exist "DynaGuide-main\results" mkdir "DynaGuide-main\results"

call conda run -n %ENV_NAME% python -c "import torch, pybullet, cv2, gym, hydra, robomimic, calvin_env, quaternion, git, einops, sklearn, diffusers; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
if errorlevel 1 goto failed

echo.
echo Done. Next: conda activate %ENV_NAME% ^&^& cd DynaGuide-main
exit /b 0

:no_conda
echo ERROR: conda not found. Install Miniconda/Anaconda and run this from an Anaconda Prompt. 1>&2
exit /b 1

:failed
echo ERROR: setup failed, see the output above. 1>&2
exit /b 1
