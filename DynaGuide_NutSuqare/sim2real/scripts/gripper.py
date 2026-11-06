import time
import pprint
import numpy as np
from scipy.spatial.transform import Rotation as R

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface


def safe_to_numpy(x):
    """
    尽量把对象转成 numpy，失败就原样返回
    """
    if x is None:
        return None
    try:
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)
    except Exception:
        return x


def rotmat_to_quat_wxyz(rot):
    """
    rot: 3x3 rotation matrix
    return: quat [w, x, y, z]
    """
    quat_xyzw = R.from_matrix(rot).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )
    return quat_wxyz


def try_print_attr(obj, name):
    """
    安全打印某个属性
    """
    print(f"\n[{name}]")
    try:
        val = getattr(obj, name)
        arr = safe_to_numpy(val)

        if isinstance(arr, np.ndarray):
            print("type:", type(val))
            print("shape:", arr.shape, "dtype:", arr.dtype)
            print(arr)
            if np.issubdtype(arr.dtype, np.floating):
                print("has_nan:", np.isnan(arr).any())
        else:
            print("type:", type(val))
            print(val)
    except Exception as e:
        print("ERROR:", repr(e))


def main():
    pp = pprint.PrettyPrinter(indent=2)

    robot_interface = FrankaInterface(
        config_root + "/charmander.yml",
        use_visualizer=False,
    )

    print("waiting for robot state...")
    while len(robot_interface._state_buffer) == 0:
        time.sleep(0.05)

    print("state stream ready")

    # 先列出所有和 gripper / pose / q 相关的候选属性
    print("\n===== candidate attrs =====")
    candidate_attrs = []
    for x in dir(robot_interface):
        lx = x.lower()
        if (
            "gripper" in lx
            or "eef" in lx
            or x in ["last_q", "last_dq", "last_eef_rot_and_pos"]
        ):
            candidate_attrs.append(x)

    for x in candidate_attrs:
        print(x)

    print("\n===== start streaming =====")
    try:
        while True:
            print("\n" + "=" * 70)

            # 1) 关节状态
            if hasattr(robot_interface, "last_q"):
                try_print_attr(robot_interface, "last_q")

            if hasattr(robot_interface, "last_dq"):
                try_print_attr(robot_interface, "last_dq")

            # 2) EEF pose
            if hasattr(robot_interface, "last_eef_rot_and_pos"):
                print("\n[last_eef_rot_and_pos]")
                try:
                    last_eef_rot, last_eef_pos = robot_interface.last_eef_rot_and_pos

                    eef_rot = np.array(last_eef_rot, dtype=np.float32)
                    if eef_rot.shape != (3, 3):
                        eef_rot = eef_rot.reshape(3, 3)

                    eef_pos = np.array(last_eef_pos, dtype=np.float32).reshape(-1)
                    if eef_pos.shape[0] > 3:
                        eef_pos = eef_pos[:3]

                    eef_quat = rotmat_to_quat_wxyz(eef_rot)

                    print("eef_rot shape:", eef_rot.shape)
                    print(eef_rot)
                    print("eef_pos shape:", eef_pos.shape)
                    print(eef_pos)
                    print("eef_quat_wxyz shape:", eef_quat.shape)
                    print(eef_quat)
                except Exception as e:
                    print("ERROR:", repr(e))

            # 3) 所有包含 gripper 的属性都试着打印
            print("\n===== gripper-related attrs =====")
            gripper_attrs = [x for x in dir(robot_interface) if "gripper" in x.lower()]
            if len(gripper_attrs) == 0:
                print("No gripper-related attributes found.")
            else:
                for x in gripper_attrs:
                    try_print_attr(robot_interface, x)

            # 4) 状态 buffer 本身也可以看一眼最后一项类型
            print("\n[_state_buffer info]")
            try:
                print("buffer length:", len(robot_interface._state_buffer))
                last_state = robot_interface._state_buffer[-1]
                print("last state type:", type(last_state))
                # 不直接 whole print，避免太长
                if hasattr(last_state, "__dict__"):
                    print("last state keys:")
                    pp.pprint(list(last_state.__dict__.keys()))
            except Exception as e:
                print("ERROR:", repr(e))

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    try:
        robot_interface.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()