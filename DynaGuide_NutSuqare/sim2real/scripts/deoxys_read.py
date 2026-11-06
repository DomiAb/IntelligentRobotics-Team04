import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface


def rotmat_to_quat_wxyz(rot):
    """
    rot: 3x3 rotation matrix
    return: quat in [w, x, y, z]
    """
    quat_xyzw = R.from_matrix(rot).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return quat_wxyz


def main():
    robot_interface = FrankaInterface(
        config_root + "/charmander.yml",
        use_visualizer=False,
    )

    print("waiting for robot state...")
    while len(robot_interface._state_buffer) == 0:
        time.sleep(0.05)

    print("state stream ready")

    # 先打印可能有用的属性，帮助你找 gripper
    print("\n===== candidate robot attrs =====")
    for x in dir(robot_interface):
        if "gripper" in x.lower() or x in ["last_q", "last_dq", "last_eef_rot_and_pos"]:
            print(x)

    try:
        while True:
            if len(robot_interface._state_buffer) == 0:
                time.sleep(0.05)
                continue

            last_eef_rot, last_eef_pos = robot_interface.last_eef_rot_and_pos

            # 整理 eef position
            eef_pos = np.array(last_eef_pos, dtype=np.float32).reshape(-1)
            if eef_pos.shape[0] > 3:
                eef_pos = eef_pos[:3]

            # 整理 eef rotation
            eef_rot = np.array(last_eef_rot, dtype=np.float32)
            if eef_rot.shape != (3, 3):
                eef_rot = eef_rot.reshape(3, 3)

            eef_quat = rotmat_to_quat_wxyz(eef_rot)

            print("\n-----")
            print("robot0_eef_pos:", eef_pos, eef_pos.shape)
            print("robot0_eef_quat:", eef_quat, eef_quat.shape)

            # 尝试读取 gripper
            gripper_val = None
            if hasattr(robot_interface, "last_gripper_q"):
                g = np.array(robot_interface.last_gripper_q, dtype=np.float32).reshape(-1)
                if g.size == 1:
                    gripper_val = np.array([g[0], g[0]], dtype=np.float32)
                elif g.size >= 2:
                    gripper_val = g[:2]

            elif hasattr(robot_interface, "last_gripper_qpos"):
                g = np.array(robot_interface.last_gripper_qpos, dtype=np.float32).reshape(-1)
                if g.size == 1:
                    gripper_val = np.array([g[0], g[0]], dtype=np.float32)
                elif g.size >= 2:
                    gripper_val = g[:2]

            if gripper_val is not None:
                print("robot0_gripper_qpos:", gripper_val, gripper_val.shape)
            else:
                print("robot0_gripper_qpos: NOT FOUND YET")

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    robot_interface.close()


if __name__ == "__main__":
    main()