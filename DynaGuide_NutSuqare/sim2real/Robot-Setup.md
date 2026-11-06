
# Phase 1: Initial Installation & Setup 
*Run these steps only once when setting up a new Workstation or NUC.*

## 1. Installing the RealSense SDK
Follow the [official instructions](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md) to enable connection to the camera. 

**Ubuntu 24.04 Fix:**
If you cannot complete step 3 of section *Install librealsense2* on Ubuntu 24.04, apply the fix described in [this issue](https://github.com/IntelRealSense/librealsense/issues/13065): 
Instead of continuing with step 3, follow [these libuvc installation instructions](https://github.com/IntelRealSense/librealsense/blob/master/doc/libuvc_installation.md).

After installation, run `realsense-viewer` to verify the camera connection.

## 2. Installation of Deoxys
Follow the primary installation instructions in the [Deoxys repository](https://github.com/UT-Austin-RPL/deoxys_control) (NOTE: the libfranka version used is `0.9.0`).

### 2.1 Fixes for NUC (Robot PC) Installation
Before running the make command:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
make -j build_franka=1
```

### 2.2 Fixes for Workstation (Desktop/Laptop) Installation
When trying to run python scripts (e.g., `python examples/reset_robot_joints.py`), you might encounter several errors. Fix them with the following steps:

* **ModuleNotFoundError for matplotlib:**
  ```bash
  pip install matplotlib
  ```
* **ModuleNotFoundError for deoxys:**
  Instead of manually exporting the `PYTHONPATH`, install the package in editable mode from the repository root (`deoxys_control`):
  ```bash
  # Ensure you are in the deoxys_control root directory
  pip install -e .
  ```
* **TypeError: Descriptors cannot be created directly:**
  This is a protobuf compatibility issue. Fix it by recompiling the proto files. 
  **NOTE:** You must be inside the `deoxys_control/deoxys` subdirectory when executing these commands:
  ```bash
  cd deoxys
  pip install grpcio-tools
  python -m grpc_tools.protoc -I=proto --python_out=proto --grpc_python_out=proto proto/franka-interface/*.proto
  python -m grpc_tools.protoc -I=proto --python_out=proto --grpc_python_out=proto proto/*.proto
  ```

---

# Phase 2: Daily Startup Routine
*Follow these steps every time you operate the robot.*

## 1. Robot and Robot Controller
1. Turn on the robot controller (Control Box).

## 2. Robot PC (NUC)
1. At startup, choose the `rt17 (real-time mode)` kernel.
2. When prompted, select `dpkg (repair broken packages)`, then select **yes** to process and **no** when asked to install new packages.
3. Log in to the `smlr_g2` user (password is the same as the username).
4. In wired settings, make sure `PCI Ethernet` is set to the robot.
5. Access the robot interface to activate FCI:
   * Open Chrome (Firefox may cause issues).
   * Enter the robot-controller-IP: `192.168.3.11`.
   * If prompted, enter `franka` as username and `frankaRSI` as password.
   * *Note: If another user used the robot beforehand, you might be prompted to request access. Follow the on-screen steps.*
6. Prepare the robot:
   * Unlock the robot.
   * Release the user-stop.
   * **Activate FCI.**
7. Configure IPs in `deoxys_control/deoxys/config/charmander.yml`:
   * `ROBOT.IP`: `192.168.3.11` (Control Box IP)
   * `NUC.IP`: `10.157.174.97` (NUC Local IP)
8. Start the low-level control programs from the `deoxys_control` root:
   ```bash
   # Arm control program
   ./auto_scripts/auto_arm.sh config/charmander.yml
   
   # Gripper control program (in a separate terminal)
   ./auto_scripts/auto_gripper.sh config/charmander.yml
   ```

## 3. Workstation (Laptops)
1. Connect the laptop to the network switch via an ethernet cable.
2. Ping the Robot PC (NUC) at `10.157.174.97` to ensure network connectivity.
3. Verify the IP configuration in your local `deoxys/config/charmander.yml` matches the NUC settings (especially `NUC.IP: 10.157.174.97`).
4. Connect the RealSense camera to your workstation.
5. Activate deoxys_control
```bash

conda activate dsrl

cd /home/smlr_g2/smlr_g2_WS25/deoxys_control/deoxys

export PYTHONPATH=/home/smlr_g2/smlr_g2_WS25/deoxys_control/deoxys:$PYTHONPATH
export PYTHONPATH=/home/smlr_g2/smlr_g2_WS25/deoxys_control/deoxys/g2_repo:$PYTHONPATH
```
6. Run your policy/control scripts.

---

# Phase 3: Scripts Reference (`sim2real/scripts/`)

Below is a summary of the key scripts available for robot testing and deployment.

## Diagnostic & Debugging Scripts

### `reset_robot_joints.py`
Moves the robot arm back to the golden initial joint positions before starting a new experiment, ensuring a consistent starting state.
```bash
python sim2real/scripts/reset_robot_joints.py
```

### `control_demo.py`
Minimal connectivity test script. Sends a small OSC_POSE action along the X-axis to verify that the Deoxys control pipeline is working correctly.
```bash
python sim2real/scripts/control_demo.py
```

### `deoxys_read.py`
Streams and prints the live robot state, including joint angles (`last_q`), EEF position/rotation/quaternion, and all gripper-related attributes. Press `Ctrl+C` to stop.
```bash
python sim2real/scripts/deoxys_read.py
```

### `gripper.py`
Dedicated script for debugging gripper state readout. Prints EEF position, orientation quaternion, and gripper opening width (`robot0_gripper_qpos`) every 0.5 seconds.
```bash
python sim2real/scripts/gripper.py
```

### `input_test.py`
Pipeline input test script (single camera, no robot control). Loads a policy checkpoint, reads frames from a RealSense camera, and verifies that obs preprocessing and encoder feature extraction work correctly. Displays the camera feed in a window.
```bash
python sim2real/scripts/input_test.py
```

---

## Deployment Scripts

### `agent_view_deployment.py`
**Single-camera deployment script (agentview only).** Loads a policy checkpoint, reads frames from a single RealSense camera in real time, infers actions from the combined image and low-dim robot state, and sends commands to the robot at 20 Hz. Supports live crop box adjustment:

| Key | Action |
|-----|--------|
| `ESC` | Quit |
| `s` | Save current crop images |
| `p` | Pause / resume robot execution |
| `z` | Send zero action (emergency stop) |
| `i/k/j/l/t/g/f/h` | Adjust crop box coordinates |

```bash
python sim2real/scripts/agent_view_deployment.py
```

### `2_views_deployment.py`
**Dual-camera deployment script (agentview + eye-in-hand).** Reads from two RealSense cameras simultaneously (distinguished by serial number), runs policy inference, and controls the robot. Crop box coordinates are automatically saved to `crop_config.json` on exit.
> **Note:** Set `agent_serial` and `wrist_serial` at the top of the script to match the serial numbers of the connected cameras.

```bash
python sim2real/scripts/2_views_deployment.py
```

### `wrist_agent_view_debug_pipeline.py`
**Dual-camera debug pipeline (no robot execution).** Similar to `2_views_deployment.py`, but focused on validating dual-camera obs construction and policy inference without sending any control commands to the robot. Use this to verify the full pipeline before live deployment.

```bash
python sim2real/scripts/wrist_agent_view_debug_pipeline.py
```

### `single_view_debug_pipeline_.py`
**Single-camera debug pipeline (no robot execution).** Uses the same obs construction logic as `agent_view_deployment.py`, with full integration of robot state reading and policy inference, but does not send control commands. Use this to validate the complete obs → action pipeline end-to-end.

```bash
python sim2real/scripts/single_view_debug_pipeline_.py
```

### `2linetest.py`
**Sim-to-real parallel test script.** Runs policy rollouts in the robosuite simulation environment and optionally mirrors the same actions to the real robot simultaneously (enabled via `--send_to_real`). Supports video recording, dataset saving, and printing a side-by-side comparison of simulated vs. real actions.
```bash
# Simulation only
python sim2real/scripts/2linetest.py --agent /path/to/ckpt.pth --n_rollouts 5

# Mirror actions to real robot
python sim2real/scripts/2linetest.py --agent /path/to/ckpt.pth --send_to_real --print_action_compare
```