"""
pinch_tracker.py - 3D Perspective-Invariant Pinch Detection Engine.

Uses true 3D metric world coordinates and palm-scale normalization
to detect thumb-index pinching robustly across all 3D hand orientations
(palm facing, turned 90° sideways, edge-on, tilted up/down).
"""

import time
import numpy as np
from typing import Optional, Tuple
from enum import Enum

from .hand_detector import Hand3DPose
from .filters import OneEuroFilter


class PinchState(Enum):
    HOVER = "HOVER"           # Fingers apart; cursor hovering
    PINCH_START = "START"     # Transitioned into pinch on this frame (pen down)
    PINCHING = "PINCHING"     # Sustained pinch (pen down)
    PINCH_STOP = "STOP"       # Transitioned out of pinch on this frame (pen up)


class PinchTracker:
    """
    3D Perspective-Invariant Pinch Tracker with Hysteresis & Adaptive Filtering.
    """

    def __init__(
        self,
        pinch_enter_distance_m: float = 0.030,  # 3.0 cm in metric 3D space
        pinch_exit_distance_m: float = 0.042,   # 4.2 cm in metric 3D space
        pinch_enter_ratio: float = 0.30,        # 30% of palm scale
        pinch_exit_ratio: float = 0.42,         # 42% of palm scale
        min_cutoff: float = 1.0,
        beta: float = 0.015
    ):
        """
        Parameters
        ----------
        pinch_enter_distance_m : float
            3D metric distance (meters) threshold to initiate a pinch.
        pinch_exit_distance_m : float
            3D metric distance (meters) threshold to release a pinch (hysteresis).
        pinch_enter_ratio : float
            Dimensionless ratio (dist_3d / palm_scale) to initiate a pinch.
        pinch_exit_ratio : float
            Dimensionless ratio to release a pinch (hysteresis).
        min_cutoff : float
            1€ filter base cutoff frequency for the pinch cursor.
        beta : float
            1€ filter speed adaptation coefficient.
        """
        self.enter_dist = pinch_enter_distance_m
        self.exit_dist = pinch_exit_distance_m
        self.enter_ratio = pinch_enter_ratio
        self.exit_ratio = pinch_exit_ratio

        self.is_pinched = False
        self.last_state = PinchState.HOVER

        # Dedicated drawing cursor filter
        self.cursor_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)
        self.last_cursor_pos: Optional[np.ndarray] = None

        # Metrics for HUD display
        self.current_dist_3d_m = 0.0
        self.current_ratio = 0.0
        self.pinch_progress = 0.0  # 0.0 (far apart) to 1.0 (touching)

    def reset(self):
        """Reset pinch tracker and cursor filter."""
        self.is_pinched = False
        self.last_state = PinchState.HOVER
        self.cursor_filter.reset()
        self.last_cursor_pos = None

    def update(self, pose: Optional[Hand3DPose], timestamp: Optional[float] = None) -> Tuple[PinchState, Optional[np.ndarray], float]:
        """
        Process hand pose and determine pinch state and smoothed cursor coordinates.

        Parameters
        ----------
        pose : Optional[Hand3DPose]
            Hand pose from detector. If None, releases pinch and resets cursor.
        timestamp : Optional[float]
            Current time in seconds.

        Returns
        -------
        Tuple[PinchState, Optional[np.ndarray], float]
            - PinchState: Current state (HOVER, PINCH_START, PINCHING, PINCH_STOP)
            - cursor_xy: Smoothed 2D pixel coordinates (x, y) or None if no hand
            - pinch_progress: Value from 0.0 (wide open) to 1.0 (fully pinched)
        """
        if timestamp is None:
            timestamp = time.time()

        if pose is None:
            prev_pinched = self.is_pinched
            self.is_pinched = False
            self.cursor_filter.reset()
            self.last_cursor_pos = None
            state = PinchState.PINCH_STOP if prev_pinched else PinchState.HOVER
            self.last_state = state
            return state, None, 0.0

        # Compute True 3D Euclidean Metric Distance (meters)
        # Invariant to 3D rotation, pitch, yaw, roll, and perspective foreshortening!
        thumb_3d = pose.thumb_tip_3d
        index_3d = pose.index_tip_3d
        dist_3d = float(np.linalg.norm(thumb_3d - index_3d))
        self.current_dist_3d_m = dist_3d

        # Anatomically Normalized Pinch Ratio
        # Invariant to hand size variation (children vs adults)
        ref_scale = pose.ref_scale
        ratio = dist_3d / ref_scale
        self.current_ratio = ratio

        # Pinch Progress: 1.0 when fully pinched, 0.0 when >= exit threshold
        progress = (self.exit_ratio - ratio) / (self.exit_ratio - self.enter_ratio + 1e-6)
        self.pinch_progress = float(np.clip(progress, 0.0, 1.0))

        # Schmitt Trigger / Hysteresis State Machine
        if not self.is_pinched:
            # Check condition to trigger pinch
            # Trigger if BOTH metric distance and ratio satisfy enter threshold
            if dist_3d <= self.enter_dist or ratio <= self.enter_ratio:
                # Extra check: make sure ratio isn't wildly inconsistent
                if ratio <= (self.enter_ratio * 1.15) and dist_3d <= (self.enter_dist * 1.2):
                    self.is_pinched = True
                    state = PinchState.PINCH_START
                else:
                    state = PinchState.HOVER
            else:
                state = PinchState.HOVER
        else:
            # We are currently pinching. Stay pinched until fingers clearly separate.
            if dist_3d >= self.exit_dist and ratio >= self.exit_ratio:
                self.is_pinched = False
                state = PinchState.PINCH_STOP
            else:
                state = PinchState.PINCHING

        self.last_state = state

        # Compute Screen Contact Point:
        # Midpoint between thumb tip and index tip in screen pixel space
        raw_contact_2d = 0.5 * (pose.thumb_tip_2d + pose.index_tip_2d)

        # Smooth cursor position using OneEuroFilter
        smoothed_cursor = self.cursor_filter.filter(raw_contact_2d, timestamp=timestamp)
        self.last_cursor_pos = smoothed_cursor

        return state, smoothed_cursor, self.pinch_progress
