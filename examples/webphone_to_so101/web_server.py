"""FastAPI WebSocket server for absolute-position teleoperation of SO-101 (5-DOF + gripper).

Receives absolute EE target coordinates from mobile web UI, clamps to
workspace bounds, solves analytical DH inverse kinematics, and returns
joint angles to the robot control loop and frontend UI.

Control model: joystick position → absolute EE XY position (not velocity).
"""

import json
import logging
import math
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
import uvicorn.config
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dh_ik import SO101DHIK, JOINT_NAMES, ARM_DOF

# ── Home EE target from neutral joint pose ──
_DEFAULT_JOINTS_DEG = SO101DHIK.get_default_joints_deg()
EE_HOME = SO101DHIK.forward_kinematics(_DEFAULT_JOINTS_DEG)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [JOYSTICK_SRV] %(message)s",
    datefmt="%H:%M:%S",
)

THIS_DIR = Path(__file__).resolve().parent
STATIC_DIR = THIS_DIR / "static"
CERT_FILE = THIS_DIR / "cert.pem"
KEY_FILE = THIS_DIR / "key.pem"

# ── Workspace bounds (meters / radians) ──
POS_BOUNDS = {"x": (-0.30, 0.50), "y": (-0.30, 0.30), "z": (0.00, 0.35)}
# Minimum radial distance from robot base center (meters) — EE must stay outside this cylinder.
# The arm cannot physically fold into itself; ~12cm is the mechanical limit.
BASE_EXCLUSION_RADIUS = 0.12
# Roll: wrist_roll joint limit is [-157°, 163°] ≈ [-2.74, 2.84] rad
RPY_BOUNDS = {
    "roll":  (-2.7, 2.7),
    "pitch": (-math.pi / 2, math.pi / 2),
    "yaw":   (-math.pi / 2, math.pi / 2),
}

DEBUG_EE_INTERVAL = 30


def _get_local_ip() -> str:
    try:
        r = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3)
        for ip in r.stdout.strip().split():
            if ip.startswith(("192.168.", "10.", "172.")):
                return ip
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _free_port(port: int) -> None:
    try:
        r = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=3)
    except Exception:
        pass


def _generate_self_signed_cert() -> None:
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
        "-days", "365", "-nodes", "-subj", "/CN=localhost",
    ], check=True, capture_output=True)
    logger.info(f"Generated cert: {CERT_FILE}")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class WebPhoneServer:
    """FastAPI server for absolute-position SO-101 teleoperation.

    Protocol: frontend sends {tx, ty, tz, tr, gear, gripper, reset_flag, enabled}
    where tx/ty/tz are absolute EE position targets (meters),
    tr is absolute roll angle (radians),
    pitch and yaw are locked to home values.
    """

    def __init__(self, host="0.0.0.0", port=4443):
        self._host = host; self._port = port
        self._app = FastAPI()
        self._lock = threading.Lock()
        self._is_connected = False
        self._msg_count = 0

        self._ee_target = EE_HOME.copy()
        self._joint_angles_deg = np.zeros(ARM_DOF, dtype=np.float64)
        self._gripper_cmd = 0
        self._prev_gripper_cmd = 0  # for edge detection
        self._gripper_pos = 0.5
        self._gear = 1
        self._enabled = False
        self._ik = SO101DHIK()
        self._init_home_ik()
        self._setup_routes()

    def _init_home_ik(self):
        self._ik._prev_q = np.radians(_DEFAULT_JOINTS_DEG).copy()
        self._ik._prev_valid = True
        pos = self._ee_target[:3].copy(); rpy = self._ee_target[3:].copy()
        r = self._ik.solve(position=pos, rpy=rpy)
        self._joint_angles_deg = r.q_deg.copy()
        logger.info(
            f"Home FK ({_DEFAULT_JOINTS_DEG}°): "
            f"EE=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},"
            f"r={rpy[0]:.2f},p={rpy[1]:.2f},y={rpy[2]:.2f}] "
            f"→ q=[{self._joint_angles_deg[0]:.1f},{self._joint_angles_deg[1]:.1f},"
            f"{self._joint_angles_deg[2]:.1f},{self._joint_angles_deg[3]:.1f},"
            f"{self._joint_angles_deg[4]:.1f}]°"
        )

    # ── Routes ──
    def _setup_routes(self):
        self._app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @self._app.get("/")
        async def index():
            return FileResponse(str(STATIC_DIR / "index.html"))

        @self._app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            logger.info("WebSocket client connected")
            with self._lock:
                self._is_connected = True
            try:
                while True:
                    data = await websocket.receive_text()
                    msg = json.loads(data)
                    resp = self._handle_message(msg)
                    if resp is not None:
                        await websocket.send_text(json.dumps(resp))
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                with self._lock:
                    self._is_connected = False

    # ── Core ──
    def _handle_message(self, msg: dict) -> dict | None:
        """Process one packet. Fields tx/ty/tz/tp/tr are absolute EE targets:
        tx,ty = XY position, tz = Z height, tp = pitch angle, tr = roll angle."""

        is_enabled = bool(msg.get("enabled") or False)
        gear = _clamp(int(msg.get("gear") or 1), 0, 2)
        gripper_cmd = int(msg.get("gripper") or 0)
        reset_flag = int(msg.get("reset_flag") or 0)

        tx = float(msg["tx"]) if "tx" in msg and msg["tx"] is not None else None
        ty = float(msg["ty"]) if "ty" in msg and msg["ty"] is not None else None
        tz = float(msg["tz"]) if "tz" in msg and msg["tz"] is not None else None
        tp = float(msg["tp"]) if "tp" in msg and msg["tp"] is not None else None
        tr = float(msg["tr"]) if "tr" in msg and msg["tr"] is not None else None

        with self._lock:
            self._enabled = is_enabled
            self._gear = gear
            self._gripper_cmd = gripper_cmd
            self._msg_count += 1

            # Reset
            if reset_flag == 1:
                self._ee_target[:] = EE_HOME.copy()
                self._ik.reset()
                self._init_home_ik()
                logger.info("Reset → home")

            # Set absolute EE target from frontend (only when enabled)
            if is_enabled:
                if tx is not None:
                    self._ee_target[0] = _clamp(tx, *POS_BOUNDS["x"])
                if ty is not None:
                    self._ee_target[1] = _clamp(ty, *POS_BOUNDS["y"])
                if tz is not None:
                    self._ee_target[2] = _clamp(tz, *POS_BOUNDS["z"])
                if tp is not None:
                    self._ee_target[4] = _clamp(tp, *RPY_BOUNDS["pitch"])
                if tr is not None:
                    self._ee_target[3] = _clamp(tr, *RPY_BOUNDS["roll"])
                # yaw stays at home value

            # Clamp all dimensions
            x = _clamp(self._ee_target[0], *POS_BOUNDS["x"])
            y = _clamp(self._ee_target[1], *POS_BOUNDS["y"])
            # Base exclusion zone: push EE outward if too close to base center
            r_xy = math.sqrt(x * x + y * y)
            if r_xy < BASE_EXCLUSION_RADIUS and r_xy > 1e-8:
                scale = BASE_EXCLUSION_RADIUS / r_xy
                x *= scale
                y *= scale
            z = _clamp(self._ee_target[2], *POS_BOUNDS["z"])
            roll  = _clamp(self._ee_target[3], *RPY_BOUNDS["roll"])
            pitch = _clamp(self._ee_target[4], *RPY_BOUNDS["pitch"])
            yaw   = _clamp(self._ee_target[5], *RPY_BOUNDS["yaw"])
            self._ee_target[:] = [x, y, z, roll, pitch, yaw]

            # Gripper: step-based (30% per tap), edge-triggered
            if gripper_cmd != self._prev_gripper_cmd:
                if gripper_cmd == 1:   # rising edge → open 30%
                    self._gripper_pos = min(1.0, self._gripper_pos + 0.30)
                elif gripper_cmd == 2: # rising edge → close 30%
                    self._gripper_pos = max(0.0, self._gripper_pos - 0.30)
                self._prev_gripper_cmd = gripper_cmd
            elif gripper_cmd == 0:
                self._prev_gripper_cmd = 0  # reset edge detector

            # IK
            pos = np.array([x, y, z], dtype=np.float64)
            rpy = np.array([roll, pitch, yaw], dtype=np.float64)
            result = self._ik.solve(position=pos, rpy=rpy)
            self._joint_angles_deg = result.q_deg.copy()

            if self._msg_count % DEBUG_EE_INTERVAL == 0:
                logger.info(
                    f"EE=[{x:+.3f},{y:+.3f},{z:+.3f} | "
                    f"R={roll:+.2f},P={pitch:+.2f},Y={yaw:+.2f}] "
                    f"gear={gear} en={is_enabled} "
                    f"q=[{self._joint_angles_deg[0]:+.1f},{self._joint_angles_deg[1]:+.1f},"
                    f"{self._joint_angles_deg[2]:+.1f},{self._joint_angles_deg[3]:+.1f},"
                    f"{self._joint_angles_deg[4]:+.1f}]°"
                )

            q = self._joint_angles_deg
            return {
                "q1": float(q[0]), "q2": float(q[1]), "q3": float(q[2]),
                "q4": float(q[3]), "q5": float(q[4]),
                "q6": float(self._gripper_pos),
                "gripper_out": self._gripper_cmd,
            }

    # ── Public ──
    @property
    def is_connected(self):
        with self._lock: return self._is_connected

    @property
    def enabled(self):
        with self._lock: return self._enabled

    def get_action(self):
        with self._lock:
            q = self._joint_angles_deg.copy()
            gp = self._gripper_pos
            gc = self._gripper_cmd
        a = {}
        for i, n in enumerate(JOINT_NAMES):
            a[f"{n}.pos"] = float(q[i])
        a["gripper.pos"] = float(gp)
        a["gripper.cmd"] = gc
        return a

    # ── Lifecycle ──
    def connect(self):
        _generate_self_signed_cert()
        local_ip = _get_local_ip()
        _free_port(self._port)

        def _run():
            config = uvicorn.Config(self._app, host=self._host, port=self._port,
                                    ssl_keyfile=str(KEY_FILE), ssl_certfile=str(CERT_FILE),
                                    log_level="warning")
            srv = uvicorn.Server(config)
            srv.install_signal_handlers = lambda: None
            self._uvicorn_server = srv
            srv.run()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        time.sleep(0.5)
        print(f"\n{'='*60}\n  SO-101 Absolute-Position Teleop\n"
              f"  Open: https://{local_ip}:{self._port}\n{'='*60}\n")

    def disconnect(self):
        if hasattr(self, '_uvicorn_server') and self._uvicorn_server:
            self._uvicorn_server.should_exit = True
