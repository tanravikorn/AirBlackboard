# dodoodle 🎨
> **3D Kinematic Hand-Tracking & Touchless Air-Canvas with MediaPipe & Adaptive Smoothing**

`dodoodle` is an air-drawing application powered by **MediaPipe Tasks** that tracks hand movements in full **3D metric space**, calculates logical hand orientation (Pitch, Yaw, Roll, Palm Normal), and enables smooth drawing on screen using a thumb-and-index pinch gesture.

---

## 🌟 Key Highlights & Engineering Features

### 1. True 3D Metric Pinch Tracking (Rotation & Distance Invariant)
* **The Problem with 2D Tracking:** In naive 2D approaches, pinch detection relies on screen pixel distance. When the user tilts their hand, rotates it sideways (edge-on), or changes distance from the webcam, perspective foreshortening causes false pinch triggers or broken lines.
* **The 3D Solution:** `dodoodle` uses MediaPipe's `world_landmarks`—a 3D coordinate space in **real-world meters** centered at the hand's root:
  $$\mathcal{D}_{3D} = \sqrt{(X_{\text{thumb}} - X_{\text{index}})^2 + (Y_{\text{thumb}} - Y_{\text{index}})^2 + (Z_{\text{thumb}} - Z_{\text{index}})^2}$$
* **Anatomical Scale Normalization:** To guarantee seamless performance across hand sizes (children vs. adults), the 3D distance is normalized by the hand's anatomical scale (wrist-to-middle-MCP distance $S_{\text{palm}}$):
  $$r_{\text{pinch}} = \frac{\mathcal{D}_{3D}}{S_{\text{palm}}}$$
  This dimensionless ratio remains **strictly invariant** whether your palm faces the camera, turns $90^\circ$ sideways, or tilts up/down.

### 2. Logical 3D Hand Orientation & Coordinate Frame
The model interprets the hand as a logical 3D articulated rigid/kinematic body:
* **Palm Center:** $C_{\text{palm}} = \frac{1}{4}(P_{\text{wrist}} + P_{\text{index\_mcp}} + P_{\text{middle\_mcp}} + P_{\text{pinky\_mcp}})$
* **Forward Axis ($\hat{y}$):** Vector from wrist to middle-finger MCP.
* **Palm Normal ($\hat{z}$):** Cross product of wrist-to-index and wrist-to-pinky vectors:
  $$\vec{n} = (P_{\text{index\_mcp}} - P_{\text{wrist}}) \times (P_{\text{pinky\_mcp}} - P_{\text{wrist}})$$
* **Transverse Axis ($\hat{x}$):** Orthogonal cross product $\hat{y} \times \hat{z}$.
* **Euler Angles (Pitch, Yaw, Roll):** Extracted from the rotation matrix $R = [\hat{x}, \hat{y}, \hat{z}]$ and visualized live via a mini 3D coordinate gimbal on screen.

### 3. Jitter-Free, Low-Latency Smoothing (1€ Filter)
Raw webcam landmark coordinates often suffer from high-frequency micro-jitter, while standard moving average filters introduce noticeable lag when drawing fast strokes.
`dodoodle` integrates the **One-Euro Filter** (Casiez et al., CHI 2012):
* **At Low Velocity (Stationary / Fine Detail):** Decreases cutoff frequency $f_c \to f_{c,\min}$ to eliminate tremor and jitter.
* **At High Velocity (Quick Swipes / Fast Strokes):** Dynamically increases cutoff frequency $f_c = f_{c,\min} + \beta |\dot{x}|$ to reduce filter lag to near zero.

### 4. Schmitt Trigger Hysteresis (Anti-Chatter)
Prevents flickering near the pinch threshold:
* **Enter Pinch (Pen Down):** $\mathcal{D}_{3D} \le 3.0\text{ cm}$ or $r_{\text{pinch}} \le 0.30$
* **Exit Pinch (Pen Up):** $\mathcal{D}_{3D} \ge 4.2\text{ cm}$ or $r_{\text{pinch}} \ge 0.42$
Once pinching begins, lines stay connected without accidental breaks until the fingers clearly separate.

### 5. Multi-Layer Canvas & Virtual Touchless Air-Toolbar
* **Virtual Top Toolbar:** Tap or hover over buttons in the air to switch colors, pick the eraser, adjust brush size, undo, or clear.
* **Canvas Modes:**
  * `OVERLAY`: Drawing alpha-blended over the mirrored webcam stream.
  * `SPLIT`: Camera feed on the left, pure digital canvas on the right.
  * `FULL_CANVAS`: Fullscreen dark/light paper canvas.
* **Anti-Aliased Stroke Engine:** Renders continuous curves with rounded caps (`cv2.LINE_AA`).

---

## 📁 Project Architecture

```
dodoodle/
├── README.md               # Documentation & mathematical formulation
├── requirements.txt        # Python package dependencies
├── run.py                  # Main executable application
├── models/
│   └── hand_landmarker.task# MediaPipe 3D Hand Landmarker model
├── src/
│   ├── __init__.py
│   ├── filters.py          # Vectorized One-Euro & Low-Pass filters
│   ├── hand_detector.py    # MediaPipe hand tracker & 3D orientation engine
│   ├── pinch_tracker.py    # 3D metric pinch detector with hysteresis
│   ├── canvas.py           # Multi-layer canvas with undo, colors & eraser
│   └── ui.py               # HUD, 3D coordinate gimbal, pinch meter & toolbar
└── tests/
    ├── __init__.py
    └── test_core.py        # Automated test suite (6 passing tests)
```

---

## 🚀 Quickstart & Installation

### 1. Set Up Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python3 run.py
```

### 3. Launch Simulated 3D Demo Mode (No Webcam Required)
To verify or demonstrate 3D rotation invariance without a camera:
```bash
python3 run.py --demo
```

---

## 🎮 Controls & Gestures

### Hand Gestures
| Gesture | Action |
| :--- | :--- |
| **Pinch (Thumb + Index)** | Start drawing continuous stroke (Pen Down) |
| **Separate Fingers (Untouch)** | Stop drawing / move cursor freely (Hover / Pen Up) |
| **Rotate Hand in 3D (XYZ)** | Fully supported! Drawing remains steady at any angle |
| **Air-Tap / Hover on Toolbar** | Select tool or color without touching keyboard |

### Keyboard Shortcuts
| Key | Action |
| :---: | :--- |
| **1 – 7** | Select color (Cyan, Green, Yellow, Magenta, Red, Blue, White) |
| **E** | Toggle Eraser mode |
| **+ / -** | Increase / decrease brush thickness |
| **Z** | Undo last stroke |
| **C** | Clear entire canvas |
| **T** | Toggle Canvas Mode (`OVERLAY` $\leftrightarrow$ `SPLIT` $\leftrightarrow$ `FULL_CANVAS`) |
| **K** | Toggle 3D hand skeleton overlay |
| **S** | Save drawing to PNG file |
| **H** | Toggle on-screen keyboard help modal |
| **Q / ESC** | Exit application |

---

## 🧪 Running Automated Tests
Run unit tests verifying 3D coordinate orthonormality, 1€ filter convergence, hysteresis state transitions, and 3D orientation invariance:

```bash
pytest tests/test_core.py -v
```
Output:
```
tests/test_core.py::test_one_euro_filter_steady_and_step PASSED
tests/test_core.py::test_hand_3d_orientation_orthonormality PASSED
tests/test_core.py::test_pinch_3d_orientation_invariance PASSED
tests/test_core.py::test_pinch_hysteresis_state_machine PASSED
tests/test_core.py::test_air_canvas_drawing_and_undo PASSED
tests/test_core.py::test_ui_overlay_toolbar_interaction PASSED
======================== 6 passed in 0.50s ========================
```
