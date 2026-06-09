"""语音控制健身 Demo 主程序。

流程:
1. 连接 SO101 Follower 机械臂
2. 启动 Vosk 语音识别后台线程（或键盘模式，无需麦克风）
3. 收到指令 → 播放对应轨迹 → 播完回到监听

用法:
  python main.py config.yaml           # 语音模式（需麦克风）
  python main.py config.yaml -k        # 键盘模式（按数字选择动作，空格停止）
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

from player import JOINT_NAMES, load_episode_actions, play_trajectory
from voice_recognizer import VoiceRecognizer

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
#  Keyboard command source (for testing without microphone)
# ═════════════════════════════════════════════════════════════════════════

class KeyboardCommandSource:
    """用终端键盘输入代替语音识别，用于无麦克风的开发测试。

    按键映射: 1=第一个动作, 2=第二个动作, ..., 空格=停止, q=退出
    """

    def __init__(self, cmd_list: list[str]) -> None:
        self._cmd_list = cmd_list
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_command(self, timeout: float = 0.05) -> str | None:
        try:
            return self._cmd_queue.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                try:
                    ch = sys.stdin.read(1)
                except Exception:
                    break
                if ch == 'q':
                    self._cmd_queue.put("stop")
                    self._running = False
                    break
                elif ch == ' ':
                    self._cmd_queue.put("stop")
                elif ch.isdigit():
                    idx = int(ch) - 1
                    if 0 <= idx < len(self._cmd_list):
                        self._cmd_queue.put(self._cmd_list[idx])
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def load_config(config_path: str | Path) -> dict:
    """加载 YAML 配置并做基本校验。"""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = ["follower", "voice", "playback"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"配置文件缺少 {key!r} 字段")
    return cfg


def _play_cmd_list(
    robot,
    cmd_list: list[dict[str, float]],
    input_src,
    fps: int,
    transition_time_s: float = 2.0,
) -> bool:
    """播放一帧或多帧命令列表到机械臂（含平滑过渡）。

    从当前关节位置插值到 cmd_list 的第一帧，然后播放完整序列。
    播放期间每帧检查停止指令。

    Returns
    -------
    bool
        True 表示完整播放完毕，False 表示被中断。
    """
    interval = 1.0 / fps
    stop_playback = False

    # ── 平滑过渡到起始帧 ──
    obs = robot.get_observation()
    current_q = np.array(
        [obs[f"{j}.pos"] for j in JOINT_NAMES], dtype=np.float32
    )
    target_q = np.array(
        [cmd_list[0][f"{j}.pos"] for j in JOINT_NAMES], dtype=np.float32
    )
    logger.info("当前关节: %s", np.round(current_q, 2))
    logger.info("目标起始: %s", np.round(target_q, 2))

    transition_steps = int(fps * transition_time_s)
    logger.info("过渡中 (%d 帧)...", transition_steps)
    for step in range(1, transition_steps + 1):
        new_cmd = input_src.get_command()
        if new_cmd == "stop":
            stop_playback = True
        if stop_playback:
            break
        alpha = step / transition_steps
        interp_q = current_q + (target_q - current_q) * alpha
        joint_cmd = {f"{j}.pos": float(interp_q[i]) for i, j in enumerate(JOINT_NAMES)}
        t0 = time.perf_counter()
        robot.send_action(joint_cmd)
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, interval - elapsed))

    if stop_playback:
        logger.info("过渡被中断")
        return False

    # ── 播放动作帧 ──
    total_frames = len(cmd_list)
    for i, joint_cmd in enumerate(cmd_list):
        new_cmd = input_src.get_command()
        if new_cmd == "stop":
            stop_playback = True
        if stop_playback:
            logger.info("播放被中断")
            return False
        t0 = time.perf_counter()
        robot.send_action(joint_cmd)
        elapsed = time.perf_counter() - t0
        sleep_time = max(0.0, interval - elapsed)
        if i < total_frames - 1 and not stop_playback:
            time.sleep(sleep_time)

    return True


def main(config_path: str = "config.yaml", keyboard_mode: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(config_path)

    # ── 1. 连接 Follower 机械臂 ──────────────────────────────────────
    logger.info("正在连接机械臂...")
    robot_cfg = SOFollowerRobotConfig(
        port=cfg["follower"]["port"],
        id=cfg["follower"]["id"],
        use_degrees=cfg["follower"]["use_degrees"],
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()
    logger.info("机械臂已连接: %s", robot)

    # ── 2. 加载轨迹与固定位姿 ────────────────────────────────────────
    tracks: dict[str, list[dict[str, float]]] = {}

    # 2a. 录制轨迹
    for name, root in cfg.get("trajectories", {}).items():
        root_path = Path(config_path).parent / root
        if root_path.exists():
            logger.info("加载轨迹: %s (%s)", name, root_path)
            from player import actions_to_robot_format
            actions = load_episode_actions(str(root_path))
            tracks[name] = actions_to_robot_format(actions)
        else:
            logger.warning("轨迹目录不存在，跳过: %s", root_path)

    # 2b. 固定位姿 → 单帧轨迹
    for name, angles in cfg.get("poses", {}).items():
        cmd = {f"{j}.pos": float(angles.get(j, 0)) for j in JOINT_NAMES}
        tracks[name] = [cmd]
        logger.info("固定位姿: %s → %s", name, cmd)

    if not tracks:
        logger.error("没有成功加载任何轨迹或位姿，请先录制")
        robot.disconnect()
        return

    # ── 3. 启动语音识别或键盘模式 ────────────────────────────────────
    voice_cmds = cfg["voice"]["commands"]  # {cmd_name: [keyword, ...]}
    cmd_names = list(voice_cmds.keys())

    # voice_cmds: {cmd_name: [keyword1, keyword2, ...]}
    if keyboard_mode:
        input_src = KeyboardCommandSource(cmd_names)
        print(f"\n{'=' * 50}")
        print("  键盘模式")
        for i, (cmd, keywords) in enumerate(voice_cmds.items(), 1):
            print(f"  [{i}] {keywords[0]}  →  {cmd}")
        print(f"  [空格] 停止  |  [q] 退出")
        print(f"{'=' * 50}\n")
    else:
        model_path = Path(config_path).parent / cfg["voice"]["model_path"]
        input_src = VoiceRecognizer(
            model_path=str(model_path),
            commands=voice_cmds,
        )
    input_src.start()

    fps = cfg["playback"]["fps"]
    transition_time_s = cfg["playback"].get("transition_time_s", 2.0)

    print("\n" + "=" * 50)
    print("  语音控制健身 Demo 已就绪")
    print(f"  机械臂: {robot}")
    print(f"  可用指令: {[v[0] for v in cfg['voice']['commands'].values()]}")
    print(f"  已加载: {list(tracks.keys())}")
    print("=" * 50 + "\n")

    current_command: str | None = None

    try:
        while True:
            cmd = input_src.get_command()

            if cmd == "stop":
                logger.info("收到停止指令")
                current_command = None
                continue

            if cmd is not None and cmd != current_command:
                current_command = cmd

                if cmd not in tracks:
                    logger.warning("未知指令: %s", cmd)
                    continue

                logger.info("执行指令: %s", cmd)
                finished = _play_cmd_list(
                    robot, tracks[cmd], input_src, fps, transition_time_s
                )
                if finished:
                    logger.info("%s 播放完成", cmd)
                current_command = None

            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        input_src.stop()
        robot.disconnect()
        logger.info("已退出")


if __name__ == "__main__":
    kb_mode = "-k" in sys.argv or "--keyboard" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    config = args[0] if args else "config.yaml"
    main(config, keyboard_mode=kb_mode)
