"""
canvas.py - High-precision air-drawing canvas with stroke smoothing, layers, and undo.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from enum import Enum


class CanvasMode(Enum):
    OVERLAY = "OVERLAY"          # Overlaid on mirrored camera feed
    SPLIT = "SPLIT"              # Side-by-side: camera left, clean canvas right
    FULL_CANVAS = "FULL_CANVAS"  # Pure digital canvas (black or white)


# Color presets in BGR format
PALETTE = {
    "cyan": (255, 230, 0),
    "green": (50, 230, 80),
    "magenta": (220, 40, 255),
    "yellow": (0, 220, 255),
    "orange": (0, 140, 255),
    "red": (40, 40, 240),
    "blue": (240, 120, 30),
    "white": (255, 255, 255),
    "eraser": (0, 0, 0)
}


class Stroke:
    """Represents a continuous drawn line stroke."""

    def __init__(self, color: Tuple[int, int, int], thickness: int, is_eraser: bool = False):
        self.color = color
        self.thickness = thickness
        self.is_eraser = is_eraser
        self.points: List[Tuple[int, int]] = []

    def add_point(self, pt: Tuple[int, int]):
        # Avoid duplicate consecutive points
        if not self.points or self.points[-1] != pt:
            self.points.append(pt)


class AirCanvas:
    """
    Air-drawing engine supporting multi-color drawing, variable brush size,
    spline/anti-aliased smoothing, undo, eraser, and multiple canvas modes.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Stroke history
        self.strokes: List[Stroke] = []
        self.current_stroke: Optional[Stroke] = None

        # Active drawing properties
        self.current_color_name = "cyan"
        self.current_color = PALETTE["cyan"]
        self.brush_size = 6
        self.is_eraser = False

        # Modes and visual theme
        self.mode = CanvasMode.OVERLAY
        self.dark_canvas = True  # Dark background when in FULL_CANVAS mode

        # Cached canvas layer (RGBA)
        self.canvas_layer = np.zeros((height, width, 4), dtype=np.uint8)
        self._need_full_redraw = True

    def set_color(self, name: str):
        """Set drawing color by name or activate eraser."""
        if name == "eraser":
            self.is_eraser = True
        elif name in PALETTE:
            self.is_eraser = False
            self.current_color_name = name
            self.current_color = PALETTE[name]

    def set_brush_size(self, size: int):
        """Set brush size clamped to [2, 40]."""
        self.brush_size = int(np.clip(size, 2, 40))

    def toggle_mode(self):
        """Cycle through OVERLAY -> SPLIT -> FULL_CANVAS."""
        modes = [CanvasMode.OVERLAY, CanvasMode.SPLIT, CanvasMode.FULL_CANVAS]
        idx = (modes.index(self.mode) + 1) % len(modes)
        self.mode = modes[idx]

    def start_stroke(self, pt: Tuple[float, float]):
        """Initiate a new stroke at given point (x, y)."""
        ipt = (int(round(pt[0])), int(round(pt[1])))
        stroke = Stroke(
            color=self.current_color,
            thickness=self.brush_size * (3 if self.is_eraser else 1),
            is_eraser=self.is_eraser
        )
        stroke.add_point(ipt)
        self.current_stroke = stroke
        self.strokes.append(stroke)
        self._draw_segment_on_layer(ipt, ipt, stroke.color, stroke.thickness, stroke.is_eraser)

    def continue_stroke(self, pt: Tuple[float, float]):
        """Continue current stroke to point (x, y)."""
        if self.current_stroke is None:
            self.start_stroke(pt)
            return

        ipt = (int(round(pt[0])), int(round(pt[1])))
        if not self.current_stroke.points:
            self.current_stroke.add_point(ipt)
            return

        prev_pt = self.current_stroke.points[-1]
        dist_sq = (ipt[0] - prev_pt[0]) ** 2 + (ipt[1] - prev_pt[1]) ** 2

        # Min distance threshold to avoid redundant points (at least 2 pixels)
        if dist_sq >= 4:
            self.current_stroke.add_point(ipt)
            self._draw_segment_on_layer(
                prev_pt, ipt,
                self.current_stroke.color,
                self.current_stroke.thickness,
                self.current_stroke.is_eraser
            )

    def end_stroke(self):
        """Finish the active stroke."""
        self.current_stroke = None

    def undo(self):
        """Undo the last stroke."""
        if self.strokes:
            self.strokes.pop()
            self.current_stroke = None
            self._redraw_full_canvas()

    def clear(self):
        """Clear all strokes."""
        self.strokes.clear()
        self.current_stroke = None
        self.canvas_layer.fill(0)
        self._need_full_redraw = False

    def _draw_segment_on_layer(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        is_eraser: bool
    ):
        """Draw an anti-aliased smoothed line segment with circular joint caps."""
        radius = max(1, thickness // 2)
        if is_eraser:
            # Eraser clears pixels by zeroing out alpha and color
            cv2.line(self.canvas_layer, p1, p2, (0, 0, 0, 0), thickness, lineType=cv2.LINE_AA)
            cv2.circle(self.canvas_layer, p2, radius, (0, 0, 0, 0), -1, lineType=cv2.LINE_AA)
        else:
            rgba = (color[0], color[1], color[2], 255)
            cv2.line(self.canvas_layer, p1, p2, rgba, thickness, lineType=cv2.LINE_AA)
            cv2.circle(self.canvas_layer, p2, radius, rgba, -1, lineType=cv2.LINE_AA)

    def _redraw_full_canvas(self):
        """Re-render all strokes from scratch."""
        self.canvas_layer.fill(0)
        for stroke in self.strokes:
            pts = stroke.points
            if not pts:
                continue
            if len(pts) == 1:
                self._draw_segment_on_layer(pts[0], pts[0], stroke.color, stroke.thickness, stroke.is_eraser)
            else:
                for i in range(len(pts) - 1):
                    self._draw_segment_on_layer(pts[i], pts[i + 1], stroke.color, stroke.thickness, stroke.is_eraser)
        self._need_full_redraw = False

    def render(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Composite drawing canvas onto camera frame according to current CanvasMode.

        Parameters
        ----------
        camera_frame : np.ndarray
            Current camera frame in BGR format (H, W, 3).

        Returns
        -------
        np.ndarray
            Rendered output frame ready for display.
        """
        h, w, _ = camera_frame.shape

        # Re-check size match
        if (w, h) != (self.width, self.height):
            self.width, self.height = w, h
            self.canvas_layer = cv2.resize(self.canvas_layer, (w, h))

        if self.mode == CanvasMode.OVERLAY:
            # Alpha composite drawing canvas onto camera feed
            alpha = self.canvas_layer[:, :, 3] / 255.0
            alpha_3d = np.dstack([alpha, alpha, alpha])
            drawing_bgr = self.canvas_layer[:, :, :3]
            composite = (1.0 - alpha_3d) * camera_frame + alpha_3d * drawing_bgr
            return composite.astype(np.uint8)

        elif self.mode == CanvasMode.FULL_CANVAS:
            # Pure digital canvas
            base_color = (25, 25, 30) if self.dark_canvas else (245, 245, 248)
            out = np.full((h, w, 3), base_color, dtype=np.uint8)
            alpha = self.canvas_layer[:, :, 3] / 255.0
            alpha_3d = np.dstack([alpha, alpha, alpha])
            drawing_bgr = self.canvas_layer[:, :, :3]
            composite = (1.0 - alpha_3d) * out + alpha_3d * drawing_bgr
            return composite.astype(np.uint8)

        elif self.mode == CanvasMode.SPLIT:
            # Side by side: Half camera, half clean canvas
            half_w = w // 2
            cam_half = cv2.resize(camera_frame, (half_w, h))

            base_color = (25, 25, 30) if self.dark_canvas else (245, 245, 248)
            canvas_bg = np.full((h, half_w, 3), base_color, dtype=np.uint8)

            # Crop or scale canvas to right half
            sub_canvas = cv2.resize(self.canvas_layer, (half_w, h))
            alpha = sub_canvas[:, :, 3] / 255.0
            alpha_3d = np.dstack([alpha, alpha, alpha])
            drawing_bgr = sub_canvas[:, :, :3]
            canvas_right = (1.0 - alpha_3d) * canvas_bg + alpha_3d * drawing_bgr

            out = np.hstack((cam_half, canvas_right.astype(np.uint8)))
            # Draw dividing line
            cv2.line(out, (half_w, 0), (half_w, h), (100, 100, 100), 2)
            return out

        return camera_frame

    def save(self, filepath: Optional[str] = None) -> str:
        """Save the artwork to a PNG file."""
        if filepath is None:
            filepath = f"dodoodle_art_{int(time.time())}.png"

        # Save transparent PNG or composited image
        base_color = (25, 25, 30) if self.dark_canvas else (255, 255, 255)
        out = np.full((self.height, self.width, 3), base_color, dtype=np.uint8)
        alpha = self.canvas_layer[:, :, 3] / 255.0
        alpha_3d = np.dstack([alpha, alpha, alpha])
        drawing_bgr = self.canvas_layer[:, :, :3]
        composite = (1.0 - alpha_3d) * out + alpha_3d * drawing_bgr

        cv2.imwrite(filepath, composite.astype(np.uint8))
        return filepath
