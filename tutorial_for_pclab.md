# First time in lab PC - Is pain. use the PC I have alreadz prepared! It is column 2, row 2, the right one when standing in the doar, the left one when looking at it. 
Rows and columns count start at 1, and from the door to the left/back, when looking INTO the room.
DynaGuide should be here: C:\Users\user\DynaGuide-main

1. Install miniconda: https://www.anaconda.com/download/success?reg=skipped-miniconda
2. install github (just google git for windows)
3. Search Anaconda Prompt in windows search bar and open it
4. Write conda cerate in the now opened terminal
5. Accept terms of use bz copy paste commands
6. conda create -n dynaguide python=3.10 -y
7. conda activate dynaguide
8. cd to folder where DynaGuide was downloaded
9. pip install -r requirements.txt
10. pip install cmake
11. then follow DGs readme instrcutions

# Recurrent user
Opn anaconda promt, go to DynaGuide repo nand activate the conda env. Installed packages should still be there!

AI needs to make an installer here!... DynaGuide has too many missing packages

## Dynaguide Links:

## Calvin DynaGuide: Immediate Example

1. Download the DynaGuide model from this [link](https://drive.google.com/file/d/1DeOnoDacXjBHgy1DGoRJpYIR8SyDy5fJ/view?usp=drive_link)
1. Download the Base policy from this [link](https://drive.google.com/file/d/1lcI_PBgFIYDsoK4T4qO7SGJJgI2lw0Kd/view?usp=drive_link)
1. Download the Guidance conditions for SWITCH_ON from this [link](https://drive.google.com/file/d/1wtEGnG87Y-imqD2MygcqbJwp_RGi7YNB/view?usp=drive_link)
1. Modify the `calvin_exp_configs_examples/switch_on.json` by changing the `pos_examples` file path to the downloaded guidance condition 
1. Make a folder called `results` in this directory 