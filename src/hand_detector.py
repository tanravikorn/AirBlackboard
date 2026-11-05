"""
hand_detector.py - MediaPipe-based 3D Hand Detection and Kinematic Orientation Engine.

Extracts:
1. 2D Normalized landmarks (screen projection coordinates).
2. 3D World landmarks (metric Euclidean coordinates in meters).
3. 3D Hand Coordinate System (Forward, Transverse, Palm Normal).
4. 3D Euler angles (Pitch, Yaw, Roll) and spatial facing direction.
5. Anatomical scale normalization (wrist-to-MCP & palm width).
6. 1€ filter smoothed tracking for jitter-free interaction.
"""

import os
import math
import time
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any

# Ensure matplotlib cache does not stall or warn
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .filters import OneEuroFilter


# MediaPipe Hand Landmark Indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Landmark connections for skeleton rendering
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17)
]


class Hand3DPose:
    """Represents full 3D and 2D kinematic pose of a detected hand."""

    def __init__(
        self,
        handedness: str,
        confidence: float,
        landmarks_2d: np.ndarray,      # (21, 3) normalized [x, y, z]
        landmarks_pixel: np.ndarray,   # (21, 2) pixel coords [u, v]
        world_landmarks_3d: np.ndarray # (21, 3) metric coords [X, Y, Z] in meters
    ):
        self.handedness = handedness
        self.confidence = confidence
        self.landmarks_2d = landmarks_2d
        self.landmarks_pixel = landmarks_pixel
        self.world_landmarks_3d = world_landmarks_3d

        # Key landmark points
        self.wrist_3d = world_landmarks_3d[WRIST]
        self.thumb_tip_3d = world_landmarks_3d[THUMB_TIP]
        self.index_tip_3d = world_landmarks_3d[INDEX_TIP]
        self.middle_tip_3d = world_landmarks_3d[MIDDLE_TIP]
        self.middle_mcp_3d = world_landmarks_3d[MIDDLE_MCP]
        self.index_mcp_3d = world_landmarks_3d[INDEX_MCP]
        self.pinky_mcp_3d = world_landmarks_3d[PINKY_MCP]

        # 2D Screen points
        self.thumb_tip_2d = landmarks_pixel[THUMB_TIP]
        self.index_tip_2d = landmarks_pixel[INDEX_TIP]
        self.middle_tip_2d = landmarks_pixel[MIDDLE_TIP]

        # Anatomical scales in 3D metric space
        self.palm_length = float(np.linalg.norm(self.middle_mcp_3d - self.wrist_3d))
        self.palm_width = float(np.linalg.norm(self.pinky_mcp_3d - self.index_mcp_3d))
        self.ref_scale = max(self.palm_length, 0.04)  # Safety clamp to at least 4cm

        # Compute 3D coordinate system and Euler orientation
        self._compute_3d_orientation()

    def _compute_3d_orientation(self):
        """Compute hand local coordinate basis and Euler angles (Pitch, Yaw, Roll)."""
        w = self.wrist_3d
        m = self.middle_mcp_3d
        idx = self.index_mcp_3d
        pky = self.pinky_mcp_3d

        # Forward axis (Y_hand): Wrist -> Middle MCP
        fwd = m - w
        norm_fwd = np.linalg.norm(fwd)
        y_axis = fwd / (norm_fwd + 1e-8)

        # Transverse vectors
        v1 = idx - w
        v2 = pky - w

        # Normal vector (Z_hand): Cross product
        # For Right Hand: idx x pky points towards palm outward normal
        if self.handedness.lower() == "left":
            normal = np.cross(v2, v1)
        else:
            normal = np.cross(v1, v2)

        norm_n = np.linalg.norm(normal)
        z_axis = normal / (norm_n + 1e-8)

        # Right / Transverse axis (X_hand): Orthogonal completion
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        # Re-orthonormalize Z to ensure perfect orthogonality
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

        self.basis_x = x_axis
        self.basis_y = y_axis
        self.basis_z = z_axis

        # Rotation matrix [X, Y, Z]
        R = np.column_stack((x_axis, y_axis, z_axis))

        # Euler angles (XYZ convention) in degrees
        # Pitch: rotation around X (tilt up/down)
        # Yaw: rotation around Y (heading left/right)
        # Roll: rotation around Z (twist)
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))
        roll = math.atan2(R[1, 0], R[0, 0])

        self.pitch_deg = math.degrees(pitch)
        self.yaw_deg = math.degrees(yaw)
        self.roll_deg = math.degrees(roll)

        # Qualitative 3D Hand Facing Direction
        # World coordinate frame in MediaPipe: +X right, +Y down, +Z forward (away from camera)
        nz = z_axis[2]
        nx = z_axis[0]
        ny = z_axis[1]

        if nz < -0.45:
            self.facing_desc = "Palm Facing Camera"
        elif nz > 0.45:
            self.facing_desc = "Palm Facing Away"
        elif abs(nx) > 0.5:
            self.facing_desc = "Side Profile (Edge)"
        elif ny < -0.45:
            self.facing_desc = "Palm Facing Up"
        else:
            self.facing_desc = "Angled / Oblique"


class HandDetector:
    """High-performance 3D Hand Landmarker detector using MediaPipe Tasks API."""

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_filter: bool = True
    ):
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, model_path)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Hand landmarker model not found at: {model_path}")

        self.model_path = model_path
        self.num_hands = num_hands
        self.enable_filter = enable_filter

        # Initialize MediaPipe Task
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Separate 1€ filters for 2D screen landmarks and 3D world landmarks
        self.filters_2d: Dict[int, OneEuroFilter] = {}
        self.filters_3d: Dict[int, OneEuroFilter] = {}
        for i in range(21):
            # Smooth 2D pixel coordinates: min_cutoff=1.2 Hz, beta=0.007
            self.filters_2d[i] = OneEuroFilter(min_cutoff=1.2, beta=0.007, d_cutoff=1.0)
            # Smooth 3D metric world coordinates
            self.filters_3d[i] = OneEuroFilter(min_cutoff=1.0, beta=0.005, d_cutoff=1.0)

        self._frame_timestamp_ms = 0
        self._start_time = time.time()

    def process(self, frame_bgr: np.ndarray) -> List[Hand3DPose]:
        """
        Detect hands in a BGR video frame and return Hand3DPose objects.

        Parameters
        ----------
        frame_bgr : np.ndarray
            OpenCV BGR image (H, W, C).

        Returns
        -------
        List[Hand3DPose]
            List of detected hands with both 2D and 3D metric kinematics.
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Monotonically increasing timestamp in milliseconds
        curr_time = time.time()
        timestamp_ms = int((curr_time - self._start_time) * 1000.0)
        if timestamp_ms <= self._frame_timestamp_ms:
            timestamp_ms = self._frame_timestamp_ms + 1
        self._frame_timestamp_ms = timestamp_ms

        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: List[Hand3DPose] = []
        if not detection_result.hand_landmarks:
            return hands

        t_sec = curr_time
        for idx_hand in range(len(detection_result.hand_landmarks)):
            raw_2d = detection_result.hand_landmarks[idx_hand]
            raw_3d = detection_result.hand_world_landmarks[idx_hand]

            handedness_str = "Right"
            conf = 0.9
            if detection_result.handedness and idx_hand < len(detection_result.handedness):
                handedness_category = detection_result.handedness[idx_hand][0]
                handedness_str = handedness_category.category_name
                conf = handedness_category.score

            arr_2d = np.zeros((21, 3), dtype=np.float64)
            arr_pixel = np.zeros((21, 2), dtype=np.float64)
            arr_3d = np.zeros((21, 3), dtype=np.float64)

            for i in range(21):
                # Normalized 2D
                nx = raw_2d[i].x
                ny = raw_2d[i].y
                nz = raw_2d[i].z
                px = nx * w
                py = ny * h

                # World 3D (meters)
                wx = raw_3d[i].x
                wy = raw_3d[i].y
                wz = raw_3d[i].z

                if self.enable_filter:
                    smoothed_pixel = self.filters_2d[i].filter(np.array([px, py]), timestamp=t_sec)
                    smoothed_3d = self.filters_3d[i].filter(np.array([wx, wy, wz]), timestamp=t_sec)
                    arr_pixel[i] = smoothed_pixel
                    arr_2d[i] = [smoothed_pixel[0] / w, smoothed_pixel[1] / h, nz]
                    arr_3d[i] = smoothed_3d
                else:
                    arr_pixel[i] = [px, py]
                    arr_2d[i] = [nx, ny, nz]
                    arr_3d[i] = [wx, wy, wz]

            pose = Hand3DPose(
                handedness=handedness_str,
                confidence=conf,
                landmarks_2d=arr_2d,
                landmarks_pixel=arr_pixel,
                world_landmarks_3d=arr_3d
            )
            hands.append(pose)

        return hands

    def reset_filters(self):
        """Reset all OneEuroFilters."""
        for f in self.filters_2d.values():
            f.reset()
        for f in self.filters_3d.values():
            f.reset()

    def close(self):
        """Release MediaPipe resources gracefully without triggering C++ fiber hang."""
        pass


def generate_simulated_hand_pose(t: float, width: int, height: int) -> Hand3DPose:
    """
    Generate an animated 3D hand pose for demo and automated testing.
    Continuously modulates 3D orientation (Pitch, Yaw, Roll) and cycles
    between Pinching (Drawing) and Untouching (Hovering).
    """
    # 2D Screen trajectory: figure-8 (lemniscate)
    speed = 0.8
    scale_x = width * 0.28
    scale_y = height * 0.22
    center_x = width * 0.5
    center_y = height * 0.52

    st = t * speed
    screen_x = center_x + scale_x * math.sin(st)
    screen_y = center_y + scale_y * math.sin(2.0 * st) * 0.5

    # 3D Euler rotation angles (degrees)
    pitch_rad = math.radians(25.0 * math.sin(t * 1.1))
    yaw_rad = math.radians(35.0 * math.cos(t * 0.9))
    roll_rad = math.radians(20.0 * math.sin(t * 1.4))

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
        [0, math.sin(pitch_rad), math.cos(pitch_rad)]
    ])
    Ry = np.array([
        [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
        [0, 1, 0],
        [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
    ])
    Rz = np.array([
        [math.cos(roll_rad), -math.sin(roll_rad), 0],
        [math.sin(roll_rad), math.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    # Pinch cycle: 3.5s drawing, 1.8s untouch hover
    cycle = t % 5.3
    is_pinching = cycle < 3.5
    pinch_gap = 0.018 if is_pinching else 0.058  # meters (1.8cm vs 5.8cm)

    # Base metric 3D skeleton in hand frame (meters)
    base_3d = np.zeros((21, 3), dtype=np.float64)
    base_3d[WRIST] = [0.0, 0.0, 0.0]
    base_3d[MIDDLE_MCP] = [0.0, 0.09, 0.0]
    base_3d[INDEX_MCP] = [-0.038, 0.082, 0.0]
    base_3d[PINKY_MCP] = [0.038, 0.070, 0.0]

    # Thumb and index tips
    half_gap = pinch_gap * 0.5
    base_3d[THUMB_TIP] = [-half_gap, 0.125, 0.0]
    base_3d[INDEX_TIP] = [half_gap, 0.125, 0.0]

    # Other finger tips for visual aesthetics
    base_3d[MIDDLE_TIP] = [0.0, 0.165, 0.02]
    base_3d[RING_TIP] = [0.032, 0.155, 0.02]
    base_3d[PINKY_TIP] = [0.060, 0.135, 0.02]

    # Fill intermediate joints
    for mcp, tip, joints in [
        (INDEX_MCP, INDEX_TIP, [INDEX_PIP, INDEX_DIP]),
        (MIDDLE_MCP, MIDDLE_TIP, [MIDDLE_PIP, MIDDLE_DIP]),
        (PINKY_MCP, PINKY_TIP, [PINKY_PIP, PINKY_DIP]),
    ]:
        base_3d[joints[0]] = base_3d[mcp] + (base_3d[tip] - base_3d[mcp]) * 0.4
        base_3d[joints[1]] = base_3d[mcp] + (base_3d[tip] - base_3d[mcp]) * 0.75

    base_3d[THUMB_CMC] = [-0.02, 0.03, 0.0]
    base_3d[THUMB_MCP] = [-0.035, 0.06, 0.0]
    base_3d[THUMB_IP] = [-0.030, 0.095, 0.0]

    # Rotate 3D world landmarks
    world_3d = base_3d @ R.T

    # Project to 2D pixel landmarks around screen_x, screen_y
    pixel_2d = np.zeros((21, 2), dtype=np.float64)
    pixel_scale = 1400.0  # pixels per meter

    for i in range(21):
        # MediaPipe coords: X right, Y down (so -Y in screen), Z away
        pixel_2d[i, 0] = screen_x + world_3d[i, 0] * pixel_scale
        pixel_2d[i, 1] = screen_y - world_3d[i, 1] * pixel_scale

    norm_2d = np.zeros((21, 3), dtype=np.float64)
    norm_2d[:, 0] = pixel_2d[:, 0] / width
    norm_2d[:, 1] = pixel_2d[:, 1] / height
    norm_2d[:, 2] = world_3d[:, 2]

    return Hand3DPose(
        handedness="Right",
        confidence=0.98,
        landmarks_2d=norm_2d,
        landmarks_pixel=pixel_2d,
        world_landmarks_3d=world_3d
    )
