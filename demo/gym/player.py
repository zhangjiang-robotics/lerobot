"""轨迹播放器：从 LeRobot dataset 读取动作序列并逐帧发送到 SO101 Follower。

录制时使用 lerobot-record 命令（不带摄像头），产出的 parquet 文件中
``action`` 列为 JointPosition 格式的 6 维数组。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# SO101 6 个关节的顺序（与 SOFollower._motors_ft 插入顺序一致）
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def load_episode_actions(dataset_root: str | Path, episode_index: int = 0) -> np.ndarray:
    """从 LeRobot dataset 中加载指定 episode 的动作序列。

    Parameters
    ----------
    dataset_root : str | Path
        LeRobot dataset 根目录（包含 data/ 和 meta/ 子目录）。
    episode_index : int
        要加载的 episode 编号，默认 0。

    Returns
    -------
    np.ndarray
        形状 ``(N, 6)`` 的动作序列，列顺序对应 JOINT_NAMES。
    """
    import pyarrow.parquet as pq
    import pandas as pd

    root = Path(dataset_root)
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"数据集 data 目录不存在: {data_dir}")

    # 收集所有 parquet 文件并按 episode_index 过滤
    frames: list[pd.DataFrame] = []
    for pq_file in sorted(data_dir.rglob("*.parquet")):
        table = pq.read_table(pq_file)
        df = table.to_pandas()
        ep_df = df[df["episode_index"] == episode_index]
        if len(ep_df) > 0:
            frames.append(ep_df)

    if not frames:
        raise ValueError(
            f"在 {data_dir} 中未找到 episode_index={episode_index} 的数据"
        )

    all_frames = pd.concat(frames, ignore_index=True)
    all_frames = all_frames.sort_values("frame_index")

    # 提取 action 列，转为 (N, 6) numpy 数组
    actions = np.stack(all_frames["action"].values).astype(np.float32)
    logger.info(
        "加载轨迹: %d 帧, shape=%s, 时长约 %.1fs",
        len(actions),
        actions.shape,
        len(actions) / 30.0,
    )
    return actions


def actions_to_robot_format(actions: np.ndarray) -> list[dict[str, float]]:
    """将 ``(N, 6)`` 动作数组转为 robot.send_action() 可用的字典列表。

    Parameters
    ----------
    actions : np.ndarray
        形状 ``(N, 6)``，列顺序见 JOINT_NAMES。

    Returns
    -------
    list[dict[str, float]]
        每个元素为 ``{"shoulder_pan.pos": val, ...}``。
    """
    out = []
    for row in actions:
        out.append({f"{jnt}.pos": float(row[i]) for i, jnt in enumerate(JOINT_NAMES)})
    return out


def play_trajectory(robot, actions: np.ndarray, fps: int = 30) -> None:
    """将动作序列逐帧发送到机械臂。

    Parameters
    ----------
    robot : SOFollower
        已连接的 SO101 Follower 实例。
    actions : np.ndarray
        形状 ``(N, 6)``。
    fps : int
        播放帧率。
    """
    cmd_list = actions_to_robot_format(actions)
    interval = 1.0 / fps
    logger.info("开始播放轨迹: %d 帧 @ %d fps", len(cmd_list), fps)

    for i, cmd in enumerate(cmd_list):
        t0 = time.perf_counter()
        robot.send_action(cmd)
        elapsed = time.perf_counter() - t0
        sleep_time = max(0.0, interval - elapsed)
        if i < len(cmd_list) - 1:
            time.sleep(sleep_time)

    logger.info("轨迹播放完成")


# ── 测试入口（需要真机连接） ───────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    actions = load_episode_actions("trajectories/fly_bird")
    fmt = actions_to_robot_format(actions)
    print(f"加载了 {len(actions)} 帧")
    print(f"首帧: {fmt[0]}")
    print(f"末帧: {fmt[-1]}")
