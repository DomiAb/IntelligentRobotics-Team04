import time
import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig


def main():
    # 1) 连接机器人
    robot_interface = FrankaInterface(
        config_root + "/charmander.yml",
        use_visualizer=False,
    )

    # 2) 读取控制器配置
    controller_cfg = YamlConfig(
        config_root + "/osc-pose-controller.yml"
    ).as_easydict()

    controller_type = "OSC_POSE"

    print("waiting for robot state...")
    while len(robot_interface._state_buffer) == 0:
        time.sleep(0.05)
    print("robot state ready")

    # 3) 先静置一下
    time.sleep(1.0)

    # 4) 发一个极小动作:
    # [dx, dy, dz, dax, day, daz, gripper]
    # 这里只沿 x 方向轻微移动，其他都不动
    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    print("sending small test action:", action)

    # 连续发几次，让机器人真的动一点点
    for _ in range(20):
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
        time.sleep(0.05)

    # 5) 再发零动作，停住
    zero_action = np.zeros(7, dtype=np.float32)
    for _ in range(10):
        robot_interface.control(
            controller_type=controller_type,
            action=zero_action,
            controller_cfg=controller_cfg,
        )
        time.sleep(0.05)

    print("done")

    try:
        robot_interface.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()