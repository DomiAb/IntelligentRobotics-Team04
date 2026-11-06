@echo off
rem One-command install of the DynaGuide env on Windows (Ubuntu: setup_dynaguide.sh).
rem Usage: run setup_dynaguide.bat from an Anaconda Prompt
setlocal

set "ENV_NAME=dynaguide"
set "ENV_FILE=dg_windows_env.yml"
cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 goto no_conda

rem pybullet has no Windows wheel on PyPI, so pip compiles it and needs the Visual Studio C++ build tools.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto install_build_tools
"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | findstr /r "." >nul
if errorlevel 1 goto install_build_tools
goto env

:install_build_tools
where winget >nul 2>nul
if errorlevel 1 goto no_build_tools
echo Installing the Visual Studio C++ build tools, this asks for admin rights...
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
if errorlevel 1 goto no_build_tools

:env
call conda env list | findstr /b /c:"%ENV_NAME% " >nul
if errorlevel 1 goto create

echo Updating existing env %ENV_NAME%...
call conda env update -n %ENV_NAME% -f %ENV_FILE%
if errorlevel 1 goto failed
goto check

:create
echo Creating env %ENV_NAME%, compiling pybullet can take a while...
call conda env create -n %ENV_NAME% -f %ENV_FILE%
if errorlevel 1 goto failed

:check
rem DynaGuide's modified local packages.
call conda run -n %ENV_NAME% python -m pip install -e ./DynaGuide-main/robomimic
if errorlevel 1 goto failed
call conda run -n %ENV_NAME% python -m pip install -e ./DynaGuide-main/calvin/calvin_env
if errorlevel 1 goto failed

if not exist "DynaGuide-main\results" mkdir "DynaGuide-main\results"

call conda run -n %ENV_NAME% python -c "import torch, pybullet, cv2, gym, hydra, robomimic, calvin_env, quaternion, git, einops, sklearn, diffusers; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
if errorlevel 1 goto failed

echo.
echo Done. Next: conda activate %ENV_NAME% ^&^& cd DynaGuide-main
exit /b 0

:no_conda
echo ERROR: conda not found. Install Miniconda/Anaconda and run this from an Anaconda Prompt. 1>&2
exit /b 1

:no_build_tools
echo ERROR: could not install the Visual Studio C++ build tools. See "If pybullet fails to build" in tutorial_for_pclab.md. 1>&2
exit /b 1

:failed
echo ERROR: setup failed, see the output above. 1>&2
exit /b 1
