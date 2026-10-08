import argparse
import os
import sys

# Resolve the absolute path of the project root directory (one level up from dyna_scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "dyna_scripts" else SCRIPT_DIR

# Ensure the root project folder is searched first for local modules like 'core'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Handle local repository overrides for robosuite and mimicgen cleanly
for local_repo in ["robosuite", "mimicgen"]:
    repo_path = os.path.join(PROJECT_ROOT, local_repo)
    if os.path.exists(repo_path) and not os.path.exists(os.path.join(repo_path, "__init__.py")):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

import json
import h5py
import imageio
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import math

import shutil 
import torchvision 
import torch
import pickle 

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS
import io 
import random 

import cv2 
import matplotlib.pyplot as plt 

# these are the core modules that create the dynaguide behavior 
from core.dynamics_models import FinalStatePredictionDino
from core.embedder_datasets import MultiviewDataset
from core.calvin_utils import generate_reset_state, check_state_difference
from core.dynaguide import calculate_classifier_guidance, calculate_position_guidance