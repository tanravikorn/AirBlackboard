"""
test_core.py - Comprehensive Unit Tests for AirBlackboard 3D Hand Tracking & Drawing Studio.
"""

import math
import numpy as np
import pytest

from src.filters import OneEuroFilter, LowPassFilter, ExponentialMovingAverage
from src.hand_detector import Hand3DPose
from src.pinch_tracker import PinchTracker, PinchState
from src.canvas import AirCanvas, CanvasMode, PALETTE
from src.ui import UIOverlay


def test_one_euro_filter_steady_and_step():
    """Verify 1€ filter smoothes jitter at low speed and reacts rapidly to steps."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)

    # Initial state
    v0 = f.filter(10.0, timestamp=0.0)
    assert abs(v0 - 10.0) < 1e-4

    # Stationary with tiny micro-jitter (low velocity)
    # The filter should heavily smooth jitter
    smoothed_jitter = []
    for i in range(1, 20):
        t = i * 0.033
        noise = 0.5 * (1 if i % 2 == 0 else -1)
        val = f.filter(10.0 + noise, timestamp=t)
        smoothed_jitter.append(val)

    # Variance of smoothed output should be much smaller than input noise variance
    assert np.std(smoothed_jitter) < 0.2

    # High velocity jump (step change)
    # The filter should rapidly track the jump due to beta parameter
    t_jump = 20 * 0.033
    v_step1 = f.filter(100.0, timestamp=t_jump)
    t_jump2 = 21 * 0.033
    v_step2 = f.filter(100.0, timestamp=t_jump2)
    assert v_step2 > 80.0  # Fast adaptation


def create_synthetic_hand_pose(
    wrist=(0.0, 0.0, 0.0),
    middle_mcp=(0.0, 0.09, 0.0),
    index_mcp=(-0.04, 0.08, 0.0),
    pinky_mcp=(0.04, 0.07, 0.0),
    thumb_tip=(-0.02, 0.12, 0.0),
    index_tip=(-0.02, 0.14, 0.0),
    rotation_matrix=None
) -> Hand3DPose:
    """Helper to create a synthetic 3D hand pose, optionally rotated by 3D matrix."""
    world_pts = np.zeros((21, 3), dtype=np.float64)
    world_pts[0] = wrist
    world_pts[9] = middle_mcp
    world_pts[5] = index_mcp
    world_pts[17] = pinky_mcp
    world_pts[4] = thumb_tip
    world_pts[8] = index_tip

    if rotation_matrix is not None:
        world_pts = world_pts @ rotation_matrix.T

    # Dummy 2D landmarks (pixels)
    pixel_pts = np.zeros((21, 2), dtype=np.float64)
    pixel_pts[4] = [300.0, 300.0]
    pixel_pts[8] = [310.0, 310.0]

    norm_2d = np.zeros((21, 3), dtype=np.float64)

    return Hand3DPose(
        handedness="Right",
        confidence=0.95,
        landmarks_2d=norm_2d,
        landmarks_pixel=pixel_pts,
        world_landmarks_3d=world_pts
    )


def test_hand_3d_orientation_orthonormality():
    """Verify 3D hand basis vectors form a strictly orthonormal coordinate frame."""
    pose = create_synthetic_hand_pose()

    bx, by, bz = pose.basis_x, pose.basis_y, pose.basis_z

    # Check unit lengths
    assert np.isclose(np.linalg.norm(bx), 1.0, atol=1e-3)
    assert np.isclose(np.linalg.norm(by), 1.0, atol=1e-3)
    assert np.isclose(np.linalg.norm(bz), 1.0, atol=1e-3)

    # Check mutual orthogonality (dot products = 0)
    assert np.isclose(np.dot(bx, by), 0.0, atol=1e-3)
    assert np.isclose(np.dot(by, bz), 0.0, atol=1e-3)
    assert np.isclose(np.dot(bx, bz), 0.0, atol=1e-3)

    # Check Euler angle calculation
    assert isinstance(pose.pitch_deg, float)
    assert isinstance(pose.yaw_deg, float)
    assert isinstance(pose.roll_deg, float)


def test_pinch_3d_orientation_invariance():
    """
    CRITICAL TEST: Verify that 3D metric pinch detection is 100% INVARIANT to
    arbitrary 3D hand orientation rotations (Pitch, Yaw, Roll).
    A naive 2D distance would fluctuate wildly or trigger false pinches,
    while our 3D metric calculation remains completely identical!
    """
    tracker = PinchTracker(pinch_enter_distance_m=0.028, pinch_exit_distance_m=0.038)

    # Pinched tips separated by 2.0 cm in metric space (should pinch)
    p_thumb = np.array([-0.02, 0.12, 0.0])
    p_index = np.array([-0.02, 0.14, 0.0])  # norm = 0.02m = 2.0cm

    # 1. Base unrotated pose
    pose_base = create_synthetic_hand_pose(thumb_tip=p_thumb, index_tip=p_index)
    state, cursor, prog = tracker.update(pose_base, timestamp=1.0)
    assert state == PinchState.PINCH_START
    assert tracker.is_pinched is True
    base_dist = tracker.current_dist_3d_m

    # 2. Test 3D rotations: 45° Pitch, 60° Yaw, 90° Roll
    def make_rot_x(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def make_rot_y(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def make_rot_z(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    R_arbitrary = make_rot_z(math.radians(90)) @ make_rot_y(math.radians(60)) @ make_rot_x(math.radians(45))

    pose_rotated = create_synthetic_hand_pose(
        thumb_tip=p_thumb,
        index_tip=p_index,
        rotation_matrix=R_arbitrary
    )

    tracker.reset()
    state_rot, _, _ = tracker.update(pose_rotated, timestamp=1.0)

    # 3D Metric distance should be mathematically identical under rotation
    assert np.isclose(tracker.current_dist_3d_m, base_dist, atol=1e-5)
    assert state_rot == PinchState.PINCH_START
    assert tracker.is_pinched is True


def test_pinch_hysteresis_state_machine():
    """Verify Schmitt trigger prevents chattering and properly tracks start/stop."""
    tracker = PinchTracker(pinch_enter_distance_m=0.028, pinch_exit_distance_m=0.038)

    # Step 1: Wide open (8cm gap)
    p_open = create_synthetic_hand_pose(thumb_tip=(-0.05, 0.12, 0.0), index_tip=(0.03, 0.12, 0.0))
    state, _, _ = tracker.update(p_open, timestamp=1.0)
    assert state == PinchState.HOVER
    assert not tracker.is_pinched

    # Step 2: Pinch together (2.0cm gap <= 2.8cm enter)
    p_pinched = create_synthetic_hand_pose(thumb_tip=(0.0, 0.12, 0.0), index_tip=(0.02, 0.12, 0.0))
    state, _, _ = tracker.update(p_pinched, timestamp=1.1)
    assert state == PinchState.PINCH_START
    assert tracker.is_pinched

    # Step 3: Sustained pinch
    state, _, _ = tracker.update(p_pinched, timestamp=1.2)
    assert state == PinchState.PINCHING
    assert tracker.is_pinched

    # Step 4: Intermediate distance (3.2cm gap: > enter_dist, but < exit_dist)
    # Hysteresis must keep it pinched!
    p_mid = create_synthetic_hand_pose(thumb_tip=(0.0, 0.12, 0.0), index_tip=(0.032, 0.12, 0.0))
    state, _, _ = tracker.update(p_mid, timestamp=1.3)
    assert state == PinchState.PINCHING
    assert tracker.is_pinched

    # Step 5: Untouch / Separate fingers (4.5cm gap >= 3.8cm exit)
    p_untouch = create_synthetic_hand_pose(thumb_tip=(0.0, 0.12, 0.0), index_tip=(0.045, 0.12, 0.0))
    state, _, _ = tracker.update(p_untouch, timestamp=1.4)
    assert state == PinchState.PINCH_STOP
    assert not tracker.is_pinched

    # Step 6: Next frame hovering
    state, _, _ = tracker.update(p_untouch, timestamp=1.5)
    assert state == PinchState.HOVER
    assert not tracker.is_pinched


def test_air_canvas_drawing_and_undo():
    """Verify drawing strokes, undo, clear, and rendering."""
    canvas = AirCanvas(width=640, height=480)

    assert len(canvas.strokes) == 0

    # Start stroke
    canvas.start_stroke((100, 100))
    assert len(canvas.strokes) == 1
    assert canvas.current_stroke is not None

    # Continue stroke
    canvas.continue_stroke((120, 120))
    canvas.continue_stroke((140, 140))
    assert len(canvas.current_stroke.points) == 3

    # End stroke
    canvas.end_stroke()
    assert canvas.current_stroke is None
    assert len(canvas.strokes) == 1

    # Second stroke
    canvas.start_stroke((200, 200))
    canvas.continue_stroke((210, 210))
    canvas.end_stroke()
    assert len(canvas.strokes) == 2

    # Undo
    canvas.undo()
    assert len(canvas.strokes) == 1

    # Clear
    canvas.clear()
    assert len(canvas.strokes) == 0

    # Eraser toggle
    assert canvas.is_eraser is False
    canvas.toggle_eraser()
    assert canvas.is_eraser is True
    canvas.toggle_eraser()
    assert canvas.is_eraser is False

    # Test rendering in all canvas modes
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for mode in [CanvasMode.OVERLAY, CanvasMode.SPLIT, CanvasMode.FULL_CANVAS]:
        canvas.mode = mode
        out = canvas.render(dummy_frame)
        assert out.shape == dummy_frame.shape


def test_gesture_eraser_pinch():
    """Verify Thumb + Middle finger pinch triggers eraser gesture."""
    tracker = PinchTracker(pinch_enter_distance_m=0.030, pinch_exit_distance_m=0.042)

    # Pose where Thumb and Middle tip are touching (1.8cm apart), while Index is open (7cm apart)
    pose = create_synthetic_hand_pose(
        thumb_tip=(-0.01, 0.12, 0.0),
        index_tip=(0.06, 0.12, 0.0)
    )
    # Set middle tip close to thumb
    pose.middle_tip_3d = np.array([0.005, 0.12, 0.0])
    pose.middle_tip_2d = np.array([305.0, 305.0])

    state, cursor, prog = tracker.update(pose, timestamp=1.0)
    assert tracker.is_pinched is True
    assert tracker.is_eraser_gesture is True
    assert state == PinchState.PINCH_START


def test_ui_overlay_toolbar_interaction():
    """Verify virtual toolbar hit-testing and action triggers."""
    canvas = AirCanvas(width=1280, height=720)
    ui = UIOverlay(width=1280, height=720)

    # First button is Cyan
    btn_cyan = ui.buttons[0]
    bx = (btn_cyan.rect[0] + btn_cyan.rect[2]) / 2.0
    by = (btn_cyan.rect[1] + btn_cyan.rect[3]) / 2.0

    # Hovering over button
    hit = ui.check_toolbar_interaction(np.array([bx, by]), PinchState.HOVER, canvas)
    assert hit is True

    # Pinch-tapping the button triggers action
    hit_pinch = ui.check_toolbar_interaction(np.array([bx, by]), PinchState.PINCH_START, canvas)
    assert hit_pinch is True
    assert canvas.current_color_name == "cyan"

    # Outside toolbar
    hit_outside = ui.check_toolbar_interaction(np.array([500.0, 500.0]), PinchState.HOVER, canvas)
    assert hit_outside is False
