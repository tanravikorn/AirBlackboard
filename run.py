#!/usr/bin/env python3
"""
run.py - Main entry point for the dodoodle 3D Hand Air-Drawing application.
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
import cv2

# Set MPLCONFIGDIR early
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from src.hand_detector import HandDetector, generate_simulated_hand_pose
from src.pinch_tracker import PinchTracker, PinchState
from src.canvas import AirCanvas, CanvasMode
from src.ui import UIOverlay


def parse_args():
    parser = argparse.ArgumentParser(description="dodoodle - 3D Hand Air Canvas")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    parser.add_argument("--model", type=str, default="models/hand_landmarker.task", help="Path to hand landmarker model")
    parser.add_argument("--no-flip", action="store_true", help="Do not horizontally mirror the camera feed")
    parser.add_argument("--min-cutoff", type=float, default=1.0, help="1€ filter min cutoff Hz (default: 1.0)")
    parser.add_argument("--beta", type=float, default=0.015, help="1€ filter beta coefficient (default: 0.015)")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic test frames (for headless/testing)")
    parser.add_argument("--demo", action="store_true", help="Run animated 3D hand drawing simulation demo")
    parser.add_argument("--fps", type=float, default=60.0, help="Target framerate (default: 60.0)")
    parser.add_argument("--save-frames", type=int, default=0, help="Save N frames and exit (for headless test/verification)")
    return parser.parse_args()


class ThreadedCamera:
    """
    Decoupled background camera capture thread.
    Prevents webcam sensor read locks from capping the rendering loop,
    enabling silky-smooth 60+ FPS tracking and spline rendering.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.ret, self.frame = cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.3)
        self.cap.release()


def init_camera(camera_idx: int = 0, width: int = 1280, height: int = 720, target_fps: float = 60.0):
    """
    Open webcam using AVFoundation on macOS or standard backend on Linux/Windows.
    Requests 60 FPS and wraps in ThreadedCamera for low-latency throughput.
    """
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    candidate_indices = [camera_idx] + [i for i in range(4) if i != camera_idx]

    for idx in candidate_indices:
        try:
            raw_cap = cv2.VideoCapture(idx, backend)
            if raw_cap.isOpened():
                raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                raw_cap.set(cv2.CAP_PROP_FPS, target_fps)

                # Read warm-up frames to ensure sensor & exposure are ready
                valid = False
                frame = None
                for _ in range(5):
                    ret, test_frame = raw_cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        valid = True
                        frame = test_frame
                        break
                    time.sleep(0.04)

                if valid:
                    actual_w = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if actual_w <= 0 or actual_h <= 0 and frame is not None:
                        actual_h, actual_w, _ = frame.shape
                    
                    threaded_cap = ThreadedCamera(raw_cap)
                    print(f"Camera successfully initialized on index {idx} ({actual_w}x{actual_h} @ requested {target_fps:.0f} FPS)!")
                    return threaded_cap, idx, actual_w, actual_h
                else:
                    raw_cap.release()
        except Exception:
            pass

    return None, -1, 0, 0


def main():
    args = parse_args()

    print("=" * 60)
    print(" dodoodle - 3D Hand Tracking & Air-Drawing Studio")
    print("=" * 60)
    print(f"Loading Hand Landmarker model: {args.model}")

    try:
        detector = HandDetector(
            model_path=args.model,
            num_hands=1,
            enable_filter=True
        )
    except Exception as e:
        print(f"Error initializing hand detector: {e}")
        sys.exit(1)

    tracker = PinchTracker(
        min_cutoff=args.min_cutoff,
        beta=args.beta
    )
    canvas = AirCanvas(width=args.width, height=args.height)
    ui = UIOverlay(width=args.width, height=args.height)

    # Initialize video capture
    cap = None
    camera_online = False
    consecutive_cam_failures = 0

    if not args.synthetic and not args.demo:
        print(f"Searching for camera (requested index: {args.camera}, target {args.fps:.0f} FPS)...")
        cap, actual_idx, actual_w, actual_h = init_camera(args.camera, args.width, args.height, target_fps=args.fps)
        if cap is not None:
            camera_online = True
            canvas.width = actual_w
            canvas.height = actual_h
            ui.width = actual_w
            ui.height = actual_h
            ui._init_toolbar()
        else:
            print("\n" + "!" * 60)
            print("WARNING: Could not connect to camera.")
            print("If running on macOS, ensure Camera permissions are granted:")
            print("  System Settings -> Privacy & Security -> Camera -> Enable for your Terminal/IDE.")
            print("You can press 'R' inside the app window at any time to reconnect.")
            print("!" * 60 + "\n")

    window_name = "dodoodle - 3D Hand Air Canvas"
    if not (args.synthetic or args.demo) or args.save_frames == 0:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps = 30.0
    frame_count = 0
    saved_count = 0

    print("\nControls:")
    print("  * Pinch Thumb + Index to Draw")
    print("  * Separate fingers to Stop Drawing")
    print("  * Change hand 3D angle (Pitch, Yaw, Roll) seamlessly")
    print("  * Key '1'-'7': Colors | 'E': Eraser | 'C': Clear | 'Z': Undo")
    print("  * Key 'T': Toggle Mode | 'R': Reconnect Camera | 'K': Skeleton | 'Q': Quit")
    print("\nReady! Starting tracking loop...\n")

    try:
        while True:
            t_now = time.time()
            dt = t_now - prev_time
            prev_time = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Capture or synthesize frame
            if camera_online and cap is not None and not args.demo and not args.synthetic:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    consecutive_cam_failures += 1
                    if consecutive_cam_failures > 20:
                        camera_online = False
                        print("Camera feed lost. Press 'R' to reconnect.")
                    continue
                consecutive_cam_failures = 0
                if not args.no_flip:
                    frame = cv2.flip(frame, 1)
            else:
                # Generate clean synthetic background
                frame = np.full((canvas.height, canvas.width, 3), (30, 30, 35), dtype=np.uint8)
                if args.demo:
                    cv2.putText(frame, "3D HAND SIMULATION DEMO (Continuous XYZ Rotation)", (40, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)

            # Ensure proper frame dimensions match canvas
            fh, fw, _ = frame.shape
            if (fw, fh) != (canvas.width, canvas.height):
                canvas.width, canvas.height = fw, fh
                ui.width, ui.height = fw, fh

            # Detect hands in 3D
            if args.demo:
                pose = generate_simulated_hand_pose(t_now, canvas.width, canvas.height)
                poses = [pose]
            else:
                poses = detector.process(frame)
                pose = poses[0] if poses else None

            # Track 3D pinch
            state, cursor_xy, progress = tracker.update(pose, timestamp=t_now)

            # Process virtual toolbar button hits
            toolbar_clicked = False
            if cursor_xy is not None:
                toolbar_clicked = ui.check_toolbar_interaction(cursor_xy, state, canvas)

            # Handle drawing stroke lifecycle if not interacting with top toolbar
            if not toolbar_clicked and cursor_xy is not None:
                force_eraser = getattr(tracker, "is_eraser_gesture", False)
                if state == PinchState.PINCH_START:
                    canvas.start_stroke(cursor_xy, force_eraser=force_eraser)
                elif state == PinchState.PINCHING:
                    canvas.continue_stroke(cursor_xy)
                elif state == PinchState.PINCH_STOP:
                    canvas.end_stroke()
            elif state == PinchState.PINCH_STOP or cursor_xy is None:
                canvas.end_stroke()

            # Render canvas composite
            display_frame = canvas.render(frame)

            # Draw HUD Overlays
            if ui.show_hud:
                ui.draw_toolbar(display_frame, canvas)
                ui.draw_bottom_bar(display_frame, fps, canvas)

                if not camera_online and not args.demo and not args.synthetic:
                    ui.draw_camera_error_card(display_frame, "Webcam device offline or permission required")

                if pose is not None:
                    ui.draw_skeleton(display_frame, pose)
                    ui.draw_3d_orientation_widget(display_frame, pose)
                    ui.draw_pinch_meter(display_frame, tracker, pose, canvas)

                ui.draw_cursor(display_frame, cursor_xy, tracker, canvas, pose)
                ui.draw_help_dialog(display_frame)

            # Display window or save frames
            if args.save_frames > 0:
                out_path = f"test_frame_{saved_count:02d}.png"
                cv2.imwrite(out_path, display_frame)
                saved_count += 1
                if saved_count >= args.save_frames:
                    print(f"Saved {saved_count} test frames. Exiting headless run.")
                    break

            try:
                cv2.imshow(window_name, display_frame)
            except Exception:
                # Running headless without display server
                pass

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:  # ESC or Q
                print("Quit key pressed.")
                break
            elif key in [ord('r'), ord('R')]:
                print("Retrying camera connection...")
                if cap is not None:
                    cap.release()
                cap, actual_idx, actual_w, actual_h = init_camera(args.camera, args.width, args.height, target_fps=args.fps)
                if cap is not None:
                    camera_online = True
                    consecutive_cam_failures = 0
                    canvas.width = actual_w
                    canvas.height = actual_h
                    ui.width = actual_w
                    ui.height = actual_h
                    ui._init_toolbar()
                    print(f"Camera reconnected on index {actual_idx} ({actual_w}x{actual_h})!")
                else:
                    print("Camera connection retry failed. Check permissions.")
            elif key in [ord('c'), ord('C')]:
                canvas.clear()
            elif key in [ord('z'), ord('Z')]:
                canvas.undo()
            elif key in [ord('e'), ord('E')]:
                canvas.toggle_eraser()
                status = "ERASER ACTIVE (Press 'E' or '1'-'7' to switch back)" if canvas.is_eraser else f"PEN ACTIVE (Color: {canvas.current_color_name.upper()})"
                print(f"[TOOL] {status}")
            elif key in [ord('t'), ord('T')]:
                canvas.toggle_mode()
            elif key in [ord('k'), ord('K')]:
                ui.show_skeleton = not ui.show_skeleton
            elif key in [ord('h'), ord('H')]:
                ui.show_help = not ui.show_help
            elif key in [ord('+'), ord('=')]:
                canvas.set_brush_size(canvas.brush_size + 2)
            elif key in [ord('-'), ord('_')]:
                canvas.set_brush_size(canvas.brush_size - 2)
            elif key in [ord('s'), ord('S')]:
                saved_file = canvas.save()
                print(f"Artwork saved to: {saved_file}")
            elif key == ord('1'):
                canvas.set_color("cyan")
            elif key == ord('2'):
                canvas.set_color("green")
            elif key == ord('3'):
                canvas.set_color("yellow")
            elif key == ord('4'):
                canvas.set_color("magenta")
            elif key == ord('5'):
                canvas.set_color("red")
            elif key == ord('6'):
                canvas.set_color("blue")
            elif key == ord('7'):
                canvas.set_color("white")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        detector.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Application finished cleanly.")
        os._exit(0)


if __name__ == "__main__":
    main()
