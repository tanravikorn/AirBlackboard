"""
ui.py - Polished HUD, Interactive Air-Toolbar, and 3D Hand Orientation Visualizer.
"""

import math
import time
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .hand_detector import Hand3DPose, HAND_CONNECTIONS
from .pinch_tracker import PinchState, PinchTracker
from .canvas import AirCanvas, PALETTE, CanvasMode


class Button:
    """Interactive virtual button for touchless air-tap interaction."""

    def __init__(self, rect: Tuple[int, int, int, int], label: str, action: str, color: Tuple[int, int, int]):
        self.rect = rect  # (x1, y1, x2, y2)
        self.label = label
        self.action = action
        self.color = color
        self.hover_start: Optional[float] = None
        self.hover_progress: float = 0.0

    def contains(self, pt: Tuple[float, float]) -> bool:
        x, y = pt
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


class UIOverlay:
    """HUD, Virtual Air-Toolbar, and 3D Pose Visualization Engine."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.show_skeleton = True
        self.show_hud = True
        self.show_help = False
        self.buttons: List[Button] = []
        self._init_toolbar()

    def _init_toolbar(self):
        """Build virtual touchless toolbar at top of screen."""
        self.buttons.clear()
        btn_h = 42
        btn_y1 = 12
        btn_y2 = btn_y1 + btn_h

        # Color buttons
        colors = [
            ("Cyan", "color_cyan", PALETTE["cyan"]),
            ("Green", "color_green", PALETTE["green"]),
            ("Yellow", "color_yellow", PALETTE["yellow"]),
            ("Magenta", "color_magenta", PALETTE["magenta"]),
            ("Red", "color_red", PALETTE["red"]),
            ("Blue", "color_blue", PALETTE["blue"]),
            ("White", "color_white", PALETTE["white"]),
        ]

        start_x = 20
        btn_w = 48
        gap = 8

        cur_x = start_x
        for label, action, color in colors:
            self.buttons.append(Button((cur_x, btn_y1, cur_x + btn_w, btn_y2), label, action, color))
            cur_x += btn_w + gap

        cur_x += 16
        # Eraser button
        self.buttons.append(Button((cur_x, btn_y1, cur_x + 68, btn_y2), "Eraser", "eraser", (80, 80, 80)))
        cur_x += 68 + gap

        # Brush size buttons
        self.buttons.append(Button((cur_x, btn_y1, cur_x + 40, btn_y2), "-", "brush_dec", (60, 60, 60)))
        cur_x += 40 + 4
        self.buttons.append(Button((cur_x, btn_y1, cur_x + 40, btn_y2), "+", "brush_inc", (60, 60, 60)))
        cur_x += 40 + gap + 16

        # Action buttons
        self.buttons.append(Button((cur_x, btn_y1, cur_x + 64, btn_y2), "Undo", "undo", (70, 70, 110)))
        cur_x += 64 + gap
        self.buttons.append(Button((cur_x, btn_y1, cur_x + 64, btn_y2), "Clear", "clear", (110, 60, 60)))
        cur_x += 64 + gap
        self.buttons.append(Button((cur_x, btn_y1, cur_x + 72, btn_y2), "Mode", "toggle_mode", (60, 90, 90)))

    def check_toolbar_interaction(
        self,
        cursor_xy: Optional[np.ndarray],
        pinch_state: PinchState,
        canvas: AirCanvas
    ) -> bool:
        """
        Check if cursor interactions hit toolbar buttons.
        Returns True if a button was activated.
        """
        if cursor_xy is None:
            return False

        cx, cy = cursor_xy
        now = time.time()

        for btn in self.buttons:
            if btn.contains((cx, cy)):
                # If pinched on button or hovered for > 0.7 seconds
                if btn.hover_start is None:
                    btn.hover_start = now
                hover_dur = now - btn.hover_start
                btn.hover_progress = min(hover_dur / 0.7, 1.0)

                if pinch_state == PinchState.PINCH_START or btn.hover_progress >= 1.0:
                    # Trigger action
                    self._execute_action(btn.action, canvas)
                    btn.hover_start = now + 0.5  # cooldown
                    btn.hover_progress = 0.0
                    return True
                return True
            else:
                btn.hover_start = None
                btn.hover_progress = 0.0

        return False

    def _execute_action(self, action: str, canvas: AirCanvas):
        """Execute toolbar button action."""
        if action.startswith("color_"):
            c_name = action.replace("color_", "")
            canvas.set_color(c_name)
        elif action == "eraser":
            canvas.set_color("eraser")
        elif action == "brush_inc":
            canvas.set_brush_size(canvas.brush_size + 2)
        elif action == "brush_dec":
            canvas.set_brush_size(canvas.brush_size - 2)
        elif action == "undo":
            canvas.undo()
        elif action == "clear":
            canvas.clear()
        elif action == "toggle_mode":
            canvas.toggle_mode()

    def draw_skeleton(self, frame: np.ndarray, pose: Hand3DPose):
        """Draw 3D hand skeleton with depth cues and joint nodes."""
        if not self.show_skeleton:
            return

        pts_2d = pose.landmarks_pixel
        z_vals = pose.world_landmarks_3d[:, 2]

        # Draw bones
        for idx1, idx2 in HAND_CONNECTIONS:
            p1 = (int(pts_2d[idx1, 0]), int(pts_2d[idx1, 1]))
            p2 = (int(pts_2d[idx2, 0]), int(pts_2d[idx2, 1]))

            # Subtle bone color gradient
            cv2.line(frame, p1, p2, (200, 200, 200), 2, lineType=cv2.LINE_AA)

        # Highlight thumb and index finger bones
        for idx1, idx2 in [(1, 2), (2, 3), (3, 4)]:
            p1 = (int(pts_2d[idx1, 0]), int(pts_2d[idx1, 1]))
            p2 = (int(pts_2d[idx2, 0]), int(pts_2d[idx2, 1]))
            cv2.line(frame, p1, p2, (255, 180, 50), 3, lineType=cv2.LINE_AA)

        for idx1, idx2 in [(5, 6), (6, 7), (7, 8)]:
            p1 = (int(pts_2d[idx1, 0]), int(pts_2d[idx1, 1]))
            p2 = (int(pts_2d[idx2, 0]), int(pts_2d[idx2, 1]))
            cv2.line(frame, p1, p2, (50, 220, 255), 3, lineType=cv2.LINE_AA)

        # Draw joints
        for i in range(21):
            px, py = int(pts_2d[i, 0]), int(pts_2d[i, 1])
            # Depth scaling: closer points appear slightly larger
            radius = int(np.clip(4 - z_vals[i] * 30, 2, 8))

            if i in [4, 8]:  # Thumb tip and index tip
                cv2.circle(frame, (px, py), radius + 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (px, py), radius + 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), radius, (80, 220, 100), -1, lineType=cv2.LINE_AA)

    def draw_3d_orientation_widget(
        self,
        frame: np.ndarray,
        pose: Hand3DPose,
        origin_xy: Tuple[int, int] = (140, 600),
        axis_len: int = 50
    ):
        """
        Draw a 3D coordinate gimbal representing the hand's logical 3D orientation.
        Red = X (transverse), Green = Y (forward), Blue = Z (palm normal).
        """
        ox, oy = origin_xy

        # Background card
        card_w, card_h = 240, 130
        x1, y1 = ox - 20, oy - 75
        x2, y2 = x1 + card_w, y1 + card_h

        # Semi-transparent dark card
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 90), 1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Local basis vectors from pose
        bx, by, bz = pose.basis_x, pose.basis_y, pose.basis_z

        # Project 3D basis vectors to 2D using isometric-like projection
        # Screen projection: X_screen = X - 0.5 * Z, Y_screen = -Y + 0.5 * Z
        def proj(v: np.ndarray) -> Tuple[int, int]:
            px = int(ox + axis_len * (v[0] - 0.35 * v[2]))
            py = int(oy + axis_len * (v[1] + 0.35 * v[2]))
            return px, py

        px_x, py_x = proj(bx)
        px_y, py_y = proj(by)
        px_z, py_z = proj(bz)

        # Draw 3D Axes:
        # X: Red (Transverse)
        cv2.arrowedLine(frame, (ox, oy), (px_x, py_x), (50, 50, 240), 2, tipLength=0.25)
        cv2.putText(frame, "X", (px_x + 4, py_x), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 255), 1)

        # Y: Green (Forward)
        cv2.arrowedLine(frame, (ox, oy), (px_y, py_y), (50, 230, 80), 2, tipLength=0.25)
        cv2.putText(frame, "Y", (px_y + 4, py_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 255, 80), 1)

        # Z: Blue (Palm Normal)
        cv2.arrowedLine(frame, (ox, oy), (px_z, py_z), (255, 180, 50), 3, tipLength=0.25)
        cv2.putText(frame, "Z (Norm)", (px_z + 4, py_z), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 80), 1)

        # Origin node
        cv2.circle(frame, (ox, oy), 4, (255, 255, 255), -1)

        # Euler angle readouts
        tx = ox + 65
        ty = y1 + 22
        cv2.putText(frame, f"Pitch: {pose.pitch_deg:+.1f} deg", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.putText(frame, f"Yaw:   {pose.yaw_deg:+.1f} deg", (tx, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.putText(frame, f"Roll:  {pose.roll_deg:+.1f} deg", (tx, ty + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # Facing description
        cv2.putText(frame, f"{pose.facing_desc}", (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 240, 255), 1)

    def draw_pinch_meter(self, frame: np.ndarray, tracker: PinchTracker, pose: Hand3DPose):
        """Draw dynamic 3D pinch proximity meter."""
        meter_x = self.width - 240
        meter_y = 70
        meter_w = 210
        meter_h = 75

        # Background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (20, 20, 25), -1)
        cv2.rectangle(overlay, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (80, 80, 90), 1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # State text
        if tracker.is_pinched:
            state_text = "PINCHING (DRAW)"
            state_color = (50, 255, 100)
        else:
            state_text = "OPEN (HOVER)"
            state_color = (200, 200, 200)

        cv2.putText(frame, state_text, (meter_x + 12, meter_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_color, 2)

        # 3D Metric readouts
        dist_cm = tracker.current_dist_3d_m * 100.0
        ratio_pct = tracker.current_ratio * 100.0
        cv2.putText(frame, f"3D Gap: {dist_cm:.1f} cm ({ratio_pct:.0f}%)", (meter_x + 12, meter_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Progress bar
        bar_x = meter_x + 12
        bar_y = meter_y + 52
        bar_w = meter_w - 24
        bar_h = 10

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 55), -1)
        fill_w = int(bar_w * tracker.pinch_progress)
        bar_color = (50, 255, 100) if tracker.is_pinched else (255, 180, 50)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    def draw_cursor(
        self,
        frame: np.ndarray,
        cursor_xy: Optional[np.ndarray],
        tracker: PinchTracker,
        canvas: AirCanvas,
        pose: Optional[Hand3DPose] = None
    ):
        """Draw smooth cursor feedback at the pinch location."""
        if cursor_xy is None:
            return

        cx, cy = int(round(cursor_xy[0])), int(round(cursor_xy[1]))
        brush_color = canvas.current_color if not canvas.is_eraser else (200, 200, 200)
        brush_r = canvas.brush_size

        if tracker.is_pinched:
            # Drawing active: compact solid dot + halo
            cv2.circle(frame, (cx, cy), brush_r + 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), brush_r, brush_color, -1, lineType=cv2.LINE_AA)
            if canvas.is_eraser:
                cv2.circle(frame, (cx, cy), brush_r * 2, (100, 100, 255), 1, lineType=cv2.LINE_AA)
        else:
            # Hover state: targeting reticle
            cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, brush_color, -1, lineType=cv2.LINE_AA)

            # Draw connector between thumb and index tips
            if pose is not None:
                p_thumb = (int(pose.thumb_tip_2d[0]), int(pose.thumb_tip_2d[1]))
                p_index = (int(pose.index_tip_2d[0]), int(pose.index_tip_2d[1]))
                cv2.line(frame, p_thumb, p_index, (150, 200, 255), 1, lineType=cv2.LINE_AA)

    def draw_toolbar(self, frame: np.ndarray, canvas: AirCanvas):
        """Render virtual interactive touchless toolbar."""
        for btn in self.buttons:
            x1, y1, x2, y2 = btn.rect
            is_active = False

            if btn.action.startswith("color_") and not canvas.is_eraser:
                c_name = btn.action.replace("color_", "")
                if c_name == canvas.current_color_name:
                    is_active = True
            elif btn.action == "eraser" and canvas.is_eraser:
                is_active = True

            # Button background
            bg_color = (40, 40, 45)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)

            # Color preview for color buttons
            if btn.action.startswith("color_"):
                cv2.rectangle(frame, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), btn.color, -1)
            else:
                # Text label
                text_size = cv2.getTextSize(btn.label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
                tx = x1 + (x2 - x1 - text_size[0]) // 2
                ty = y1 + (y2 - y1 + text_size[1]) // 2
                cv2.putText(frame, btn.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1)

            # Active border
            border_color = (0, 255, 255) if is_active else (100, 100, 110)
            thickness = 2 if is_active else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness)

            # Hover progress fill indicator
            if btn.hover_progress > 0.0:
                fill_h = int((y2 - y1) * btn.hover_progress)
                cv2.rectangle(frame, (x1, y2 - fill_h), (x2, y2), (0, 255, 200), 2)

    def draw_bottom_bar(self, frame: np.ndarray, fps: float, canvas: AirCanvas):
        """Draw bottom status information bar."""
        bar_h = 32
        y1 = self.height - bar_h
        cv2.rectangle(frame, (0, y1), (self.width, self.height), (18, 18, 22), -1)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (14, y1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 255, 180), 1)

        # Brush Info
        brush_info = f"Brush: {canvas.brush_size}px" if not canvas.is_eraser else "ERASER"
        cv2.putText(frame, brush_info, (100, y1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

        # Mode Info
        mode_text = f"Mode: {canvas.mode.value}"
        cv2.putText(frame, mode_text, (240, y1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 220, 255), 1)

        # Key shortcuts guide
        shortcuts = "[Pinch] Draw | [C] Clear | [Z] Undo | [E] Eraser | [T] Mode | [R] Camera | [H] Help | [Q] Quit"
        text_size = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        cv2.putText(frame, shortcuts, (self.width - text_size[0] - 14, y1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 180), 1)

    def draw_camera_error_card(self, frame: np.ndarray, message: str = "Webcam not detected or unauthorized"):
        """Draw prominent troubleshooting card when camera is inaccessible."""
        cx, cy = self.width // 2, self.height // 2
        w, h = 540, 240
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 30), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 120, 240), 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        lines = [
            "! WEBCAM NOT DETECTED / ACCESS NEEDED",
            "",
            f"Status: {message}",
            "",
            "TO ENABLE ON MACOS:",
            "  1. Open System Settings -> Privacy & Security -> Camera",
            "  2. Enable Camera permission for Terminal / Python / Your IDE",
            "  3. Press 'R' on your keyboard to retry connecting",
            "  4. Or run: python3 run.py --demo (to test without camera)"
        ]

        ty = y1 + 32
        for line in lines:
            if line.startswith("!"):
                color = (0, 140, 255)
            elif line.startswith("TO ENABLE"):
                color = (0, 240, 255)
            elif line.startswith("  3. Press"):
                color = (50, 255, 100)
            else:
                color = (220, 220, 220)
            cv2.putText(frame, line, (x1 + 24, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
            ty += 22

    def draw_help_dialog(self, frame: np.ndarray):
        """Draw interactive keyboard help modal."""
        if not self.show_help:
            return

        cx, cy = self.width // 2, self.height // 2
        w, h = 520, 340
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 30), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (120, 120, 140), 2)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

        lines = [
            "--- DODUSTUDIO / DODODLE HELP ---",
            "",
            "GESTURES:",
            "  * Pinch (Thumb + Index): Draw line",
            "  * Untouch / Separate: Stop drawing",
            "  * Change 3D Hand Orientation: Fully supported!",
            "",
            "KEYBOARD SHORTCUTS:",
            "  * 1 - 7: Quick color select",
            "  * E: Toggle Eraser",
            "  * + / -: Increase / Decrease brush size",
            "  * Z: Undo last stroke",
            "  * C: Clear canvas",
            "  * T: Toggle canvas mode (Overlay / Split / Full)",
            "  * R: Reconnect / Re-detect camera",
            "  * K: Toggle 3D Skeleton overlay",
            "  * S: Save drawing to PNG",
            "  * H: Toggle this help screen",
            "  * Q / ESC: Exit application"
        ]

        ty = y1 + 30
        for line in lines:
            color = (0, 255, 255) if line.startswith("---") or line.endswith(":") else (230, 230, 230)
            cv2.putText(frame, line, (x1 + 24, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
            ty += 21
