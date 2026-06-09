"""DH analytical inverse kinematics for SO-ARM-101 (5-DOF arm).

Solves the 2R planar problem for the arm, then matches orientation.
Selects the solution closest to the previous joint configuration
to avoid discontinuities (smooth following).

Geometric parameters extracted from URDF: SO101/so101_new_calib.urdf
"""

import math
from dataclasses import dataclass, field

import numpy as np

# ── SO-101 geometric parameters (extracted from URDF) ──
# Measured from joint centers at q=0 (arm vertical, pointing up)
_DH = {
    "d1":  0.0624,   # shoulder_pan base height (m)
    "a2":  0.1120,   # upper arm: shoulder_lift → elbow_flex along arm (m)
    "a3":  0.1350,   # forearm:  elbow_flex → wrist_flex along arm (m)
    "d5":  0.0980,   # tool: wrist_roll → gripper_frame along roll axis (m)
}
# Joint limits from URDF
_JOINT_LIMITS_DEG = {
    "shoulder_pan":  (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex":    (-96.8,  96.8),
    "wrist_flex":    (-95.0,  95.0),
    "wrist_roll":    (-157.2, 162.8),
}
_JOINT_LIMITS_RAD = {
    k: (math.radians(lo), math.radians(hi))
    for k, (lo, hi) in _JOINT_LIMITS_DEG.items()
}
JOINT_NAMES = list(_JOINT_LIMITS_DEG.keys())
ARM_DOF = 5


@dataclass
class IKSolution:
    """One IK solution candidate."""
    q: np.ndarray       # joint angles [q1..q5] (radians)
    valid: bool = True  # within limits?


@dataclass
class IKResult:
    """Full IK output."""
    q_deg: np.ndarray       # target joint angles [5] (degrees)
    over_limit: bool = False  # True if target is outside workspace
    solution_index: int = 0   # which solution was chosen
    position_error: float = 0.0  # residual position error (m)


class SO101DHIK:
    """DH analytical inverse kinematics for SO-101.

    Algorithm:
        Step 1 — shoulder_pan:  θ1 = atan2(y_target, x_target)
        Step 2 — 2R planar problem: law of cosines for shoulder_lift & elbow_flex
        Step 3 — wrist_flex:       angle from planar geometry to match desired pitch
        Step 4 — wrist_roll:       directly from desired orientation roll

        Up to 4 solutions (θ1 ± π, elbow up/down).
        "Nearest solution" selects the candidate with minimum L2 distance
        to the previous joint configuration.
    """

    def __init__(self):
        self._prev_q = np.zeros(ARM_DOF, dtype=np.float64)  # radians
        self._prev_valid = False
        # Per-joint max step per frame (rad) for smooth following
        self._max_joint_step_rad = math.radians(8.0)

    def reset(self) -> None:
        """Clear previous solution (called after re-zero)."""
        self._prev_valid = False

    def solve(
        self,
        position: np.ndarray,   # [x, y, z] meters in robot base frame
        rpy: np.ndarray,        # [roll, pitch, yaw] radians
        scale: float = 1.0,
    ) -> IKResult:
        """Compute IK for a 6DOF target.

        Args:
            position: Target position [x, y, z] in robot base frame (meters).
            rpy:      Target orientation [roll, pitch, yaw] (radians).
            scale:    Position scaling factor (phone_motion * scale = robot_motion).

        Returns:
            IKResult with joint angles or over_limit flag.
        """
        # Scale position
        x = position[0] * scale
        y = position[1] * scale
        z = position[2] * scale
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

        d1 = _DH["d1"]
        a2 = _DH["a2"]
        a3 = _DH["a3"]
        d5 = _DH["d5"]

        # ---- Step 1: shoulder_pan ----
        r_proj = math.sqrt(x * x + y * y)
        if r_proj < 1e-8:
            # Target directly above base → any θ1 is valid, keep previous
            candidates_theta1 = [self._prev_q[0], -self._prev_q[0]]
        else:
            theta1_a = math.atan2(y, x)          # 0° → forward
            theta1_b = math.atan2(-y, -x)         # ±180°
            candidates_theta1 = [theta1_a, theta1_b]

        # ---- Step 2: 2R planar (shoulder_lift, elbow_flex) ----
        z_shoulder = z - d1  # height relative to shoulder
        r2 = r_proj * r_proj + z_shoulder * z_shoulder
        max_reach = a2 + a3
        min_reach = abs(a2 - a3)

        # Reachability check → clamp to boundary instead of failing
        r = math.sqrt(r2)
        if r > max_reach and r_proj > 1e-8:
            # Clamp to max reach
            ratio = max_reach / r
            r_proj *= ratio
            z_shoulder *= ratio
        elif r < min_reach and r_proj > 1e-8:
            # Clamp to min reach
            ratio = min_reach / max(r, 1e-6)
            r_proj *= ratio
            z_shoulder *= ratio

        # Law of cosines for elbow
        cos_theta3 = (r_proj * r_proj + z_shoulder * z_shoulder - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
        cos_theta3 = max(-1.0, min(1.0, cos_theta3))
        sin_theta3_up   = math.sqrt(max(0.0, 1.0 - cos_theta3 * cos_theta3))   # elbow up
        sin_theta3_down = -sin_theta3_up                                         # elbow down

        candidates = []

        for theta1 in candidates_theta1:
            for sin_theta3 in (sin_theta3_up, sin_theta3_down):
                theta3 = math.atan2(sin_theta3, cos_theta3)

                # Solve theta2
                # θ2 = atan2(z_shoulder, r_proj) - atan2(a3*sin(θ3), a2 + a3*cos(θ3))
                psi = math.atan2(z_shoulder, r_proj) if r_proj > 1e-8 else 0.0
                phi = math.atan2(a3 * math.sin(theta3), a2 + a3 * math.cos(theta3))
                theta2 = psi - phi

                # Step 3: wrist_flex
                # Total pitch from shoulder_lift: actual pitch = theta2 + theta3 + theta4
                theta4 = pitch - theta2 - theta3

                # Step 4: wrist_roll (directly from roll)
                theta5 = roll

                q = np.array([theta1, theta2, theta3, theta4, theta5])

                # Check joint limits
                valid = True
                for i, name in enumerate(JOINT_NAMES):
                    lo, hi = _JOINT_LIMITS_RAD[name]
                    if q[i] < lo - 0.001 or q[i] > hi + 0.001:
                        valid = False
                        break

                if valid:
                    candidates.append(IKSolution(q=q, valid=True))

        # ---- Select nearest to previous configuration ----
        if not candidates:
            # All solutions violate joint limits → use previous q (best effort)
            if self._prev_valid:
                best = self._prev_q.copy()
            else:
                best = np.zeros(ARM_DOF, dtype=np.float64)
            self._prev_q = best.copy()
            self._prev_valid = False
            return IKResult(
                q_deg=np.degrees(best),
                over_limit=True,
                position_error=0.0,
            )

        if self._prev_valid and candidates:
            # Choose candidate closest to previous q
            best_idx = 0
            best_dist = float("inf")
            for idx, sol in enumerate(candidates):
                dist = float(np.sum((sol.q - self._prev_q) ** 2))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        else:
            # First solution: prefer elbow-up (more natural)
            # Elbow-up means θ3 > 0
            best_idx = 0
            for idx, sol in enumerate(candidates):
                if sol.q[2] > 0:  # elbow_flex positive = elbow up
                    best_idx = idx
                    break

        best = candidates[best_idx].q.copy()

        # Apply per-joint max step limit (anti-jerk)
        if self._prev_valid:
            for i in range(ARM_DOF):
                step = best[i] - self._prev_q[i]
                if abs(step) > self._max_joint_step_rad:
                    best[i] = self._prev_q[i] + math.copysign(
                        self._max_joint_step_rad, step
                    )

        # Clamp to hard limits
        for i, name in enumerate(JOINT_NAMES):
            lo, hi = _JOINT_LIMITS_RAD[name]
            best[i] = max(lo, min(hi, best[i]))

        self._prev_q = best.copy()
        self._prev_valid = True

        # Convert to degrees
        q_deg = np.degrees(best)

        return IKResult(
            q_deg=q_deg,
            over_limit=False,
            solution_index=len(candidates),
            position_error=0.0,
        )

    @staticmethod
    def get_joint_limits_deg() -> dict:
        """Return joint limits as {name: (lo_deg, hi_deg)}."""
        return dict(_JOINT_LIMITS_DEG)

    @staticmethod
    def forward_kinematics(q_deg: np.ndarray) -> np.ndarray:
        """Compute end-effector pose from joint angles (matching IK convention).

        Args:
            q_deg: [θ1, θ2, θ3, θ4, θ5] in degrees
                   θ1=shoulder_pan, θ2=shoulder_lift, θ3=elbow_flex,
                   θ4=wrist_flex, θ5=wrist_roll

        Returns:
            [x, y, z, roll, pitch, yaw] (meters, radians)
        """
        θ1 = math.radians(q_deg[0])
        θ2 = math.radians(q_deg[1])
        θ3 = math.radians(q_deg[2])
        θ4 = math.radians(q_deg[3])
        θ5 = math.radians(q_deg[4])

        d1 = _DH["d1"]
        a2 = _DH["a2"]
        a3 = _DH["a3"]
        d5 = _DH["d5"]

        # Wrist center in arm-plane (before d5)
        wr = a2 * math.cos(θ2) + a3 * math.cos(θ2 + θ3)
        wz = a2 * math.sin(θ2) + a3 * math.sin(θ2 + θ3)

        # Tool pitch = sum of lift + elbow + wrist
        pitch = θ2 + θ3 + θ4

        # EE position
        x = wr * math.cos(θ1) + d5 * math.cos(pitch) * math.cos(θ1)
        y = wr * math.sin(θ1) + d5 * math.cos(pitch) * math.sin(θ1)
        z = d1 + wz + d5 * math.sin(pitch)

        return np.array([
            x, y, z,
            θ5,     # roll
            pitch,  # pitch
            θ1,     # yaw
        ])

    @staticmethod
    def get_default_joints_deg() -> np.ndarray:
        """Return default/neutral joint angles (degrees) for SO-101.

        Based on MuJoCo simulation reset pose and leisaac calibration rest-pose centers.
        This is the pose the arm should be in when the controller is at its zero point.
        """
        return np.array([0.0, -20.0, 60.0, -70.0, 0.0], dtype=np.float64)

    @property
    def prev_q_deg(self) -> np.ndarray:
        return np.degrees(self._prev_q)
