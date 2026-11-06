# First time in lab PC - Is pain. use the PC I have alreadz prepared! It is column 2, row 2, the right one when standing in the doar, the left one when looking at it. 
Rows and columns count start at 1, and from the door to the left/back, when looking INTO the room.
DynaGuide should be here: C:\Users\user\DynaGuide-main

## Install on a new Windows PC

You need an NVIDIA GPU with a current driver, because the code runs on CUDA.

1. Install miniconda: https://www.anaconda.com/download/success?reg=skipped-miniconda
2. Install git (just google git for windows) and clone this repo
3. Search Anaconda Prompt in windows search bar and open it
4. Accept the terms of use of the conda channels:
   ```bat
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
   ```
5. Go to the repo root and run the setup script once:
   ```bat
   cd path\to\IntelligentRobotics-Team04
   setup_dynaguide.bat
   ```

The script installs the Visual Studio C++ build tools if they are missing (it asks for admin rights), creates the conda env `dynaguide` from `dg_windows_env.yml`, installs the local `robomimic` and `calvin_env`, and creates the `results` folder. At the end it prints the torch version and `CUDA available: True`. It is safe to run again after a failure.

This is what the script does for the env. To do it by hand, without the build tools check:

```bat
conda env create -f dg_windows_env.yml
conda activate dynaguide
```

### If pybullet fails to build

`pybullet==3.2.7` has no Windows wheel on PyPI, so pip compiles it. The usual error is `Microsoft Visual C++ 14.0 or greater is required`. Install the build tools, then run `setup_dynaguide.bat` again:

```bat
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

If winget says the build tools are already installed, the C++ part is missing. Open the Visual Studio Installer, click Modify on Build Tools 2022, tick "Desktop development with C++" and install.

If winget does not exist on the PC, download the Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and tick the same option.

### Speed up a PC that already has the env

Do this once. `setup_dynaguide.bat` does the same on a new PC. pip builds pybullet without numpy, which makes the cameras slower, and this rebuilds it with numpy. The physics code doesn't change:

```bat
conda activate dynaguide
pip install --force-reinstall --no-deps --no-cache-dir --no-build-isolation pybullet==3.2.7
python -c "import pybullet; print('numpy enabled:', pybullet.isNumpyEnabled())"
```

The last line must print `numpy enabled: 1`. If the rebuild fails, your old pybullet stays installed and everything still works, only slower.

# Recurrent user
Opn anaconda promt, go to DynaGuide repo nand activate the conda env. Installed packages should still be there!

```bat
conda activate dynaguide
cd DynaGuide-main
```

## Dynaguide Links:

To run Dynaguide, open the Notebook [DynaGuide_Experiments](DynaGuide-main/DynaGuide_Experiments.ipynb)
