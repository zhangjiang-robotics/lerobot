"""读取 SO101 机械臂当前关节位置，以 YAML 格式输出（用于粘贴到 config.yaml）。"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    robot = SO101Follower(SOFollowerRobotConfig(
        port=cfg["follower"]["port"],
        id=cfg["follower"]["id"],
        use_degrees=cfg["follower"]["use_degrees"],
    ))
    robot.connect()
    obs = robot.get_observation()

    print("\n复制以下内容到 config.yaml 的 poses: 下：\n")
    for j in JOINT_NAMES:
        val = obs[f"{j}.pos"]
        print(f"      {j}: {val:.2f}")

    robot.disconnect()

if __name__ == "__main__":
    main()
