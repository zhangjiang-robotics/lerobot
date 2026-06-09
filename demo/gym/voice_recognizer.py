"""Vosk 离线中文语音识别模块。

后台线程持续监听麦克风，匹配预设关键词，将匹配结果放入队列供主线程消费。
支持 PulseAudio 和 PyAudio 两种后端，自动选择可用的。
每个指令支持多个同义词关键词。
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceRecognizer:
    """离线中文语音识别器，基于 Vosk。

    自动选择音频后端:
    1. PulseAudio (parec) — 适用于 ALSA 不完整的 conda 环境
    2. PyAudio — 标准方案

    Parameters
    ----------
    model_path : str | Path
        Vosk 中文模型目录路径。
    commands : dict[str, list[str]]
        命令名 → 关键词列表。如 ``{"fly_bird": ["飞鸟", "鸟"], "stop": ["停止", "停"]}``。
    sample_rate : int
        音频采样率 (Hz)，默认 16000。
    """

    def __init__(
        self,
        model_path: str | Path,
        commands: dict[str, list[str]],
        sample_rate: int = 16000,
    ) -> None:
        import vosk

        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Vosk 模型未找到: {model_path}\n"
                f"下载: wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip\n"
                f"解压到 demo/gym/ 目录下"
            )
        self._model = vosk.Model(model_path)
        self._commands = commands
        self._all_keywords = [kw for kws in commands.values() for kw in kws]
        self._sample_rate = sample_rate
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_partial = ""

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("语音识别已启动，关键词: %s", self._all_keywords)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_command(self, timeout: float = 0.05) -> str | None:
        try:
            return self._cmd_queue.get_nowait()
        except queue.Empty:
            return None

    # ── 音频读取后端 ──────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._run_pulseaudio()
        except Exception as e:
            logger.warning("PulseAudio 不可用 (%s)，尝试 PyAudio...", e)
            try:
                self._run_pyaudio()
            except Exception as e2:
                logger.error("PyAudio 也不可用: %s", e2)
                self._running = False

    def _run_pulseaudio(self) -> None:
        import vosk

        proc = subprocess.Popen(
            [
                "parec",
                "--format=s16le",
                f"--rate={self._sample_rate}",
                "--channels=1",
                "--latency-msec=50",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        logger.info("音频后端: PulseAudio (parec)")

        recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
        recognizer.SetWords(True)

        chunk_bytes = 4000 * 2

        while self._running:
            try:
                data = proc.stdout.read(chunk_bytes)
            except Exception:
                time.sleep(0.1)
                continue
            if not data:
                time.sleep(0.05)
                continue

            # 完整句子
            if recognizer.AcceptWaveform(data):
                self._parse_result(recognizer.Result())
            else:
                # 部分结果（实时片段）
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "").strip()
                if text and text != self._last_partial:
                    self._last_partial = text
                    self._match_keywords(text, source="partial")

        proc.terminate()
        proc.wait(timeout=2)

    def _run_pyaudio(self) -> None:
        import vosk
        import pyaudio

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16, channels=1,
            rate=self._sample_rate, input=True, frames_per_buffer=8000,
        )
        logger.info("音频后端: PyAudio")
        recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
        recognizer.SetWords(True)
        stream.start_stream()

        while self._running:
            try:
                data = stream.read(4000, exception_on_overflow=False)
            except Exception:
                time.sleep(0.1)
                continue
            if recognizer.AcceptWaveform(data):
                self._parse_result(recognizer.Result())
            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "").strip()
                if text and text != self._last_partial:
                    self._last_partial = text
                    self._match_keywords(text, source="partial")

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def _parse_result(self, result_json: str) -> None:
        result = json.loads(result_json)
        text = result.get("text", "").strip()
        if text:
            logger.info("识别: %s", text)
            self._match_keywords(text, source="final")
        self._last_partial = ""

    def _match_keywords(self, text: str, source: str) -> None:
        for cmd_name, keywords in self._commands.items():
            for kw in keywords:
                if kw in text:
                    logger.info("指令: %s (%s) → %s", kw, source, cmd_name)
                    self._cmd_queue.put(cmd_name)
                    return


# ── 测试入口 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    vr = VoiceRecognizer(
        model_path="vosk-model-small-cn-0.22",
        commands={
            "fly_bird": ["飞鸟", "飞"],
            "shoulder_press": ["推肩", "肩推", "肩"],
            "face_pull": ["面拉", "拉面", "拉"],
            "go_home": ["归零", "回家", "归位"],
            "stop": ["停止", "停", "停下"],
        },
    )
    vr.start()
    print("\n说中文指令... (Ctrl+C 退出)\n")
    try:
        while True:
            cmd = vr.get_command(timeout=0.2)
            if cmd:
                print(f">>> {cmd}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        vr.stop()
