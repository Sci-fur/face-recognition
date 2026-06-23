# Plan: Face Recognition System — Cross-Platform Standalone Standard

## Vision

Transform the current CLI app into a **portable, layered system** where the core engine
is shared across multiple frontends — starting with a **Linux Desktop GUI (Priority 1)**,
then a **Web App (Priority 2)**, while keeping the original **CLI** fully intact.

The guiding principle: **one core, many surfaces.**

```
┌────────────────────────────────────────────────────────────┐
│                  SURFACE LAYER (Frontends)                 │
│  ┌──────────┐    ┌───────────────────┐   ┌──────────────┐  │
│  │  CLI     │    │  Desktop GUI      │   │  Web App     │  │
│  │(current) │    │  Linux .AppImage  │   │  (Future)    │  │
│  └────┬─────┘    └────────┬──────────┘   └──────┬───────┘  │
│       └─────────────────┬─┘─────────────────────┘          │
│                         ▼                                  │
│               SERVICE / API LAYER                          │
│          FaceRecognitionEngine  (engine.py)                │
│                         │                                  │
│        ┌────────────────┼────────────────┐                 │
│        ▼                ▼                ▼                 │
│   detector.py      embedder.py     classifier.py           │
└────────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Refactor: Extract the Core Engine
> **This is a hard prerequisite. Do not start Phase 1 until Phase 0 is complete and tested.**

The entire ML pipeline must be decoupled from CLI I/O into a single importable class.
`engine.py` becomes the contract that every frontend talks to. No frontend ever imports
`collect.py`, `train.py`, or `recognize.py` directly.

### 0.1 — Design the `FaceRecognitionEngine` API

- [ ] Create `engine.py` with class `FaceRecognitionEngine`
- [ ] Constructor `__init__(self, dataset_dir="dataset", model_dir="model")`:
  - Store paths as instance attributes
  - Instantiate `Detector`, `Embedder` (lazy — only when first needed, not at import time)
  - Set `self.model_loaded = False`
- [ ] Method `load_model(self) → bool`:
  - Load `face_svm.pkl`, `scaler.pkl`, `classes.npy` from `model_dir`
  - Return `True` on success, `False` if files missing
  - Set `self.model_loaded = True`
  - Must be safe to call multiple times (idempotent)
- [ ] Method `get_dataset_info(self) → dict`:
  - Return `{"persons": [{"name": str, "count": int}, ...], "total_images": int}`
  - Return empty persons list if `dataset_dir` doesn't exist
- [ ] Method `collect_frame(self, frame: np.ndarray, name: str) → dict`:
  - Run MTCNN detection on frame
  - If face detected with confidence ≥ 0.95: save to `dataset/{name}/img_{n:04d}.jpg`, return
    `{"saved": True, "count": int, "face_bbox": [x1,y1,x2,y2], "stage_hint": str}`
  - If no face: return `{"saved": False, "count": int, "face_bbox": None, "stage_hint": str}`
  - `stage_hint` is computed from current count (every ~17 images advances the 9 pose stages)
- [ ] Method `train(self, progress_callback=None) → dict`:
  - Accept optional `progress_callback(step: str, pct: int)` callable for GUI progress bars
  - Call it at: dataset scan (10%), embedding extraction per-person (10–80%), SVM fit (85%), save (95%), done (100%)
  - Return `{"success": bool, "train_acc": float, "val_acc": float, "n_persons": int, "n_images": int, "error": str|None}`
- [ ] Method `recognize_frame(self, frame: np.ndarray) → list[dict]`:
  - Return `[]` if `self.model_loaded` is False (caller must handle gracefully)
  - For each detected face return:
    `{"name": str, "confidence": float, "is_unknown": bool, "bbox": [x1,y1,x2,y2]}`
  - Apply existing centroid-distance rejection and 60% probability threshold
- [ ] Method `delete_person(self, name: str) → bool`:
  - Remove `dataset/{name}/` directory
  - Return `True` on success, `False` if not found
  - Does NOT retrain — caller is responsible for retraining after

### 0.2 — Thin out the CLI modules

- [ ] Rewrite `collect.py` to use `engine.collect_frame()` in its loop:
  - Keep all `cv2.imshow`, `print`, and keyboard handling here
  - Remove direct imports of `Detector` — go through engine
- [ ] Rewrite `train.py` to call `engine.train(progress_callback=lambda s,p: print(f"[{p}%] {s}"))`:
  - Keep all `print` status output here
- [ ] Rewrite `recognize.py` to call `engine.recognize_frame()` in its loop:
  - Keep all `cv2.imshow`, bounding box drawing, FPS display here
- [ ] Verify `app.py` needs no changes (it calls collect/train/recognize modules unchanged)

### 0.3 — Regression tests

- [ ] Run `./run.sh collect --name "TestPerson"` → confirm images saved to `dataset/TestPerson/`
- [ ] Run `./run.sh train` → confirm model files appear in `model/`, accuracy printed
- [ ] Run `./run.sh recognize` → confirm live feed with labels works
- [ ] Confirm all three commands produce identical behavior to before the refactor

**Deliverables:** `engine.py`, updated `collect.py`, `train.py`, `recognize.py`. Zero behavior change on CLI.

---

## Phase 1 — Linux Desktop GUI App (Priority 1)

### Technology: PyQt6 + AppImage packaging

**Why PyQt6 for Linux:**
- Installs via pip, no system Qt required
- Native GTK-like look on most Linux DEs (GNOME, KDE, XFCE)
- `QThread` + signals/slots is the correct pattern for background camera work
- `QPixmap.fromImage()` renders OpenCV frames natively in a `QLabel`
- PyInstaller produces a single-folder build; `appimagetool` wraps it into a portable `.AppImage`
- No Electron, no Node.js, no browser — pure Python

**Why NOT alternatives:**
- `tkinter` — no native threading model, ugly on modern Linux
- `wxPython` — complex install on Linux, inconsistent look
- `Kivy` — mobile-focused, poor desktop UX
- Electron/Tauri — introduces JS/Node dependency, overkill for local app

### 1.1 — App Layout Design

```
┌─────────────────────────────────────────────────────────────────┐
│  🎥 Face Recognition System                        [─] [□] [×]  │
├─────────────────┬───────────────────────────────────────────────┤
│ SIDEBAR (220px) │  CAMERA FEED (fills remaining width)          │
│                 │                                               │
│  MODE           │   ┌─────────────────────────────────────┐    │
│  ◉ Recognize    │   │                                     │    │
│  ○ Collect      │   │     Live video frame here           │    │
│  ○ Train        │   │     (bounding boxes + labels)       │    │
│                 │   │                                     │    │
│  ─────────────  │   └─────────────────────────────────────┘    │
│  KNOWN PERSONS  │                                               │
│  Alice    (150) │  ──────────────────────────────────────────   │
│  Bob      (150) │  LOG / STATUS (120px tall, scrollable)        │
│  Saifur   (150) │  [12:03:01] Model loaded — 3 persons          │
│                 │  [12:03:05] Alice recognized — 94.2%          │
│  [+ Add Person] │  [12:03:06] Unknown face detected             │
│  [🗑 Delete]    │                              FPS: 12          │
│  [⚙ Train]     │                                               │
└─────────────────┴───────────────────────────────────────────────┘
```

### 1.2 — Setup

- [ ] Add `PyQt6` to `requirements.txt`
- [ ] Add `pyinstaller` and `appimagetool` notes to `requirements-dev.txt`
- [ ] Create `gui/` package: `mkdir gui && touch gui/__init__.py`
- [ ] Create `desktop_app.py` at project root (entry point: `python desktop_app.py`):
  - Create `QApplication`
  - Instantiate `FaceRecognitionEngine`
  - Call `engine.load_model()` at startup (silently — no crash if missing)
  - Show `MainWindow`
  - Exit cleanly: stop camera thread on window close

### 1.3 — `gui/camera_thread.py`

This is the most critical piece. The camera loop MUST run off the main thread or the UI freezes.

- [ ] Create `CameraThread(QThread)`:
  - Signals: `frame_ready = pyqtSignal(np.ndarray)`, `result_ready = pyqtSignal(list)`, `error = pyqtSignal(str)`
  - Constructor takes `engine: FaceRecognitionEngine` and `mode_ref: list` (a 1-element list holding current mode string — mutable so the thread always reads the latest mode without restart)
  - `run(self)` loop:
    - Open `cv2.VideoCapture(0)` — emit `error("Camera not found")` and return if it fails
    - Loop:
      - Read frame; on read failure emit `error` and break
      - Emit `frame_ready(frame)` every frame (for display)
      - Every 2nd frame, check `mode_ref[0]`:
        - `"recognize"` → call `engine.recognize_frame(frame)`, emit `result_ready(results)`
        - `"collect"` → call `engine.collect_frame(frame, collect_name_ref[0])`, emit `result_ready([result])`
        - `"train"` → do nothing (training runs in a separate `TrainWorker` thread)
      - Sleep `1ms` to yield to Qt event loop
    - Release camera on exit
- [ ] `stop(self)`:
  - Set a `self._running = False` flag; the loop checks this and exits cleanly
- [ ] Constructor also takes `collect_name_ref: list` (1-element, holds current person name during collection)

### 1.4 — `gui/main_window.py`

- [ ] Create `MainWindow(QMainWindow)`:
  - `__init__`: call `_build_ui()`, `_connect_signals()`, `_start_camera()`
- [ ] `_build_ui(self)`:
  - `QSplitter` (horizontal) as central widget with sidebar (fixed 220px) and right panel
  - **Sidebar:**
    - `QButtonGroup` with three `QRadioButton`: Recognize / Collect / Train — default Recognize
    - `QListWidget` for known persons — populated from `engine.get_dataset_info()`
    - Each item shows `"{name}  ({count})"` — use monospace for alignment
    - `[+ Add Person]` → opens `CollectDialog`
    - `[🗑 Delete]` → confirms + calls `engine.delete_person()` + refreshes list
    - `[⚙ Train]` → opens `TrainDialog`
  - **Right panel** (`QVBoxLayout`):
    - `QLabel` (`self.video_label`) with `setAlignment(Qt.AlignCenter)`, `setMinimumSize(640, 480)`
    - `QTextEdit` (`self.log_panel`) read-only, 120px tall, auto-scrolls to bottom on new message
    - FPS label (`QLabel`) right-aligned, overlaid or below feed
- [ ] `_connect_signals(self)`:
  - `camera_thread.frame_ready` → `self._on_frame(frame)`
  - `camera_thread.result_ready` → `self._on_results(results)`
  - `camera_thread.error` → `self._on_camera_error(msg)`
  - Mode radio buttons `toggled` → update `self._mode_ref[0]`
- [ ] `_on_frame(self, frame)`:
  - Convert `frame` (BGR numpy) → `QImage` → `QPixmap`
  - Scale pixmap to fit `video_label` preserving aspect ratio
  - Set on `video_label`
  - Update FPS counter (compute from time delta between calls)
- [ ] `_on_results(self, results)`:
  - If mode is `"recognize"`: draw bounding boxes on the frame before display:
    - Green box + `"{name} {confidence:.0%}"` for known faces
    - Red box + `"Unknown"` for unknown faces
  - If mode is `"collect"`: draw detection box + stage hint text overlay
  - Append to log panel for notable events (new recognition, unknown face)
- [ ] `_on_camera_error(self, msg)`:
  - Show `QMessageBox.critical` with the error
  - Disable camera-dependent controls
- [ ] `refresh_person_list(self)`:
  - Call `engine.get_dataset_info()` and repopulate `QListWidget`
  - Called after collect finishes and after delete
- [ ] `closeEvent(self, event)`:
  - Call `camera_thread.stop()`, `camera_thread.wait()` before accepting the close

### 1.5 — `gui/collect_dialog.py`

- [ ] Create `CollectDialog(QDialog)`:
  - Constructor takes `engine` and `camera_thread` (to switch its mode temporarily)
  - Layout: `QLineEdit` for name → confirm → live camera `QLabel` → `QProgressBar` (0–150) → stage hint `QLabel` → `[Done]` / `[Cancel]`
  - On name confirmed: set `camera_thread.collect_name_ref[0] = name`, set mode to `"collect"`
  - `camera_thread.result_ready` connected to `self._on_collect_result(result)`
- [ ] `_on_collect_result(self, results)`:
  - Extract the single collect result dict
  - Update progress bar to `result["count"]`
  - Update stage hint label to `result["stage_hint"]`
  - Draw detection box overlay on the frame
  - When count reaches 150: auto-close dialog, emit `collection_done` signal
- [ ] On close (Done/Cancel/X): switch camera mode back to `"recognize"`, disconnect signal

### 1.6 — `gui/train_dialog.py`

- [ ] Create `TrainDialog(QDialog)`:
  - Show dataset summary from `engine.get_dataset_info()` in a `QLabel` before training starts
  - `[Start Training]` button → disables itself, starts `TrainWorker`
  - `QProgressBar` (0–100)
  - Status `QLabel` showing current step
  - Results area (hidden until done): train acc, val acc, persons, images
  - `[Close]` button (disabled until training finishes or fails)
- [ ] Create `TrainWorker(QThread)`:
  - Signals: `progress = pyqtSignal(str, int)`, `finished = pyqtSignal(dict)`
  - `run(self)`: call `engine.train(progress_callback=self._cb)`
  - `_cb(self, step, pct)`: emit `progress(step, pct)`
  - On return: emit `finished(result_dict)`
- [ ] Connect `TrainWorker.progress` → update progress bar + status label
- [ ] Connect `TrainWorker.finished` → show results, reload model in engine (`engine.load_model()`), enable Close
- [ ] On successful train: emit `train_done` signal so `MainWindow` can refresh person list

### 1.7 — Frame Annotation Helper

- [ ] Create `gui/utils.py`:
  - `draw_results(frame: np.ndarray, results: list[dict]) → np.ndarray`:
    - Draws all bounding boxes and labels onto a copy of the frame
    - Green box + white label for known; red box + white label for Unknown
    - Font: `cv2.FONT_HERSHEY_SIMPLEX`, size 0.6, thickness 2
    - Returns annotated copy (never mutates original)
  - `numpy_to_pixmap(frame: np.ndarray) → QPixmap`:
    - `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` → `QImage` → `QPixmap`
    - Encapsulates the conversion boilerplate used in multiple places

### 1.8 — Integration test (manual)

- [ ] Launch `python desktop_app.py` with no model trained → app opens, no crash, shows "No model loaded" in log
- [ ] Click `[+ Add Person]`, enter name, collect 150 frames → progress bar fills, dialog closes
- [ ] Click `[⚙ Train]` → training dialog shows progress, completes, shows accuracy
- [ ] Main window switches to Recognize mode → live recognition works with bounding boxes
- [ ] Click `[🗑 Delete]` on a person → confirms, person disappears from list
- [ ] Close window → no hang, process exits cleanly

### 1.9 — Linux Packaging as `.AppImage`

- [ ] Add `pyinstaller` to `requirements-dev.txt`
- [ ] Create `desktop_app.spec` (PyInstaller spec file):
  - `datas`: include `model/` dir, `dataset/` dir (empty placeholder), facenet-pytorch cached weights (`~/.cache/torch/checkpoints/`)
  - `hiddenimports`: `['facenet_pytorch', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs']` (common sklearn hidden imports)
  - `excludes`: `['matplotlib', 'IPython', 'jupyter']` to reduce size
  - `onedir` mode (not onefile — PyTorch is too large for single-file to be practical)
- [ ] Create `build.sh`:
  ```bash
  #!/bin/bash
  set -e
  source venv/bin/activate
  pyinstaller desktop_app.spec --clean
  # Download appimagetool if not present
  # Wrap dist/desktop_app/ into FaceRecognition.AppImage
  ```
- [ ] Create `AppDir/` structure for appimagetool:
  - `AppDir/usr/bin/` → symlink to PyInstaller output
  - `AppDir/FaceRecognition.desktop` → `.desktop` file with Name, Exec, Icon, Categories
  - `AppDir/icon.png` → 256×256 app icon (create a simple one with Python/Pillow if needed)
- [ ] Run `build.sh` and verify `FaceRecognition.AppImage` launches on a clean Ubuntu install
- [ ] Test AppImage on Ubuntu 22.04 and Ubuntu 24.04

**New files:** `desktop_app.py`, `desktop_app.spec`, `build.sh`, `gui/__init__.py`, `gui/main_window.py`, `gui/camera_thread.py`, `gui/collect_dialog.py`, `gui/train_dialog.py`, `gui/utils.py`, `AppDir/FaceRecognition.desktop`

---

## Phase 2 — Web App (Priority 2, Future)

### Technology: FastAPI + WebSocket + plain HTML/JS

**Why FastAPI:**
- Shares `engine.py` directly — zero duplication of ML logic
- `asyncio` + WebSocket handles streaming video frames to a browser
- REST endpoints for control actions
- Runs locally (localhost:8000) or can be deployed to a server
- No Node.js, no build step

### Architecture

```
Browser (plain HTML + JS)
    │
    ├── WebSocket /ws/stream  ← annotated JPEG frames as base64, ~10fps
    ├── POST /collect/start   ← {name: str}
    ├── POST /collect/stop
    ├── POST /train           ← returns job id
    ├── GET  /train/status    ← {step, pct, done, result}
    ├── GET  /persons         ← [{name, count}]
    └── DELETE /persons/{name}
         │
         ▼
    FastAPI  (web_server.py)
         │
         ▼
    FaceRecognitionEngine (engine.py)   ← same as CLI and Desktop
```

### 2.1 — `web/web_server.py` (deferred)

- [ ] FastAPI app with lifespan: start camera capture thread on startup, stop on shutdown
- [ ] `GET /` → serve `web/index.html`
- [ ] `WebSocket /ws/stream`:
  - Background thread reads frames, calls `engine.recognize_frame()` or `engine.collect_frame()`
  - Calls `draw_results()` (reuse from `gui/utils.py` or move to `core/utils.py`)
  - Encodes frame as JPEG, sends as base64 JSON `{"frame": "...", "results": [...]}`
  - On disconnect: stop gracefully
- [ ] `POST /collect/start` → set mode to collect, set name
- [ ] `POST /collect/stop` → set mode back to recognize
- [ ] `POST /train` → start `TrainWorker` in background thread, return `{"job_id": "train_1"}`
- [ ] `GET /train/status` → return current progress from shared state dict
- [ ] `GET /persons` → `engine.get_dataset_info()`
- [ ] `DELETE /persons/{name}` → `engine.delete_person(name)`

### 2.2 — `web/index.html` (deferred)

- [ ] Single HTML file, no build step, vanilla JS only
- [ ] `<img id="feed">` updated by WebSocket messages (set `src` to `data:image/jpeg;base64,...`)
- [ ] Mode toggle buttons (Recognize / Collect)
- [ ] Add person form (name input + Start/Stop Collection)
- [ ] Train button with progress bar (polls `/train/status`)
- [ ] Person list (fetched from `/persons` on load)
- [ ] Log panel (appended from WebSocket result messages)

### 2.3 — Setup (deferred)

- [ ] Add `fastapi`, `uvicorn[standard]`, `python-multipart` to `requirements-web.txt`
- [ ] Entry point: `python web/web_server.py` → prints `http://localhost:8000`
- [ ] Add `run_web.sh` convenience launcher

---

## Final Project Structure (after all phases)

```
FaceRecognition/
│
├── app.py                    # CLI entry point (unchanged)
├── desktop_app.py            # Linux desktop GUI entry point
├── engine.py                 # Core engine — single source of truth
│
├── collect.py                # CLI collect loop (uses engine)
├── train.py                  # CLI train (uses engine)
├── recognize.py              # CLI recognize loop (uses engine)
│
├── detector.py               # MTCNN — untouched
├── embedder.py               # InceptionResnetV1 — untouched
├── classifier.py             # SVM + centroid — untouched
│
├── gui/                      # Desktop GUI (Phase 1)
│   ├── __init__.py
│   ├── main_window.py
│   ├── camera_thread.py
│   ├── collect_dialog.py
│   ├── train_dialog.py
│   └── utils.py
│
├── web/                      # Web app (Phase 2 — future)
│   ├── web_server.py
│   └── index.html
│
├── AppDir/                   # AppImage packaging assets
│   ├── FaceRecognition.desktop
│   └── icon.png
│
├── dataset/                  # Auto-generated
├── model/                    # Auto-generated
│
├── SRS.md
├── README.md
├── requirements.txt          # Core + PyQt6
├── requirements-dev.txt      # pyinstaller, appimagetool
├── requirements-web.txt      # fastapi, uvicorn (future)
├── run.sh                    # CLI launcher (unchanged)
├── build.sh                  # Linux AppImage build script
└── desktop_app.spec          # PyInstaller spec
```

---

## Dependencies

**`requirements.txt`** (runtime, all platforms):
```
# Core ML — unchanged
torch
facenet-pytorch
opencv-python
numpy
scikit-learn
Pillow
joblib

# Desktop GUI — Linux Phase 1
PyQt6
```

**`requirements-dev.txt`** (packaging only, not shipped):
```
pyinstaller>=6.0
# appimagetool installed separately as binary from github.com/AppImage/appimagetool
```

**`requirements-web.txt`** (future Phase 2):
```
fastapi
uvicorn[standard]
python-multipart
```

---

## Implementation Order

| Step | Task | Touches | Blocking |
|------|------|---------|---------|
| 1 | Design `engine.py` API (all methods, all return shapes) | `engine.py` | Nothing |
| 2 | Implement `engine.py` | `engine.py` | Step 1 |
| 3 | Thin out `collect.py`, `train.py`, `recognize.py` | 3 files | Step 2 |
| 4 | CLI regression test | all CLI | Step 3 |
| 5 | `gui/utils.py` (frame drawing + numpy→QPixmap) | `gui/utils.py` | Step 2 |
| 6 | `gui/camera_thread.py` | `gui/camera_thread.py` | Step 2, 5 |
| 7 | `desktop_app.py` + `gui/main_window.py` skeleton (static UI, no logic) | 2 files | Step 6 |
| 8 | Wire `CameraThread` to `video_label` — get live feed showing | `main_window.py` | Step 7 |
| 9 | Wire recognize results to bounding box overlay | `main_window.py` | Step 8 |
| 10 | `gui/train_dialog.py` + `TrainWorker` | `train_dialog.py` | Step 2 |
| 11 | `gui/collect_dialog.py` | `collect_dialog.py` | Step 6 |
| 12 | Wire sidebar buttons (Add Person, Delete, Train) to dialogs | `main_window.py` | Step 10, 11 |
| 13 | Manual integration test of full GUI workflow | all gui | Step 12 |
| 14 | `desktop_app.spec` + `build.sh` + AppImage packaging | packaging files | Step 13 |
| 15 | AppImage test on clean Ubuntu VM | AppImage | Step 14 |
| 16 | Phase 2: Web app (deferred) | `web/` | Step 4 |

Steps 1–4 can be done in one sitting. Steps 5–9 form the first GUI milestone (live feed working).
Steps 10–12 complete the GUI. Steps 14–15 are packaging — done last, once the app is stable.

---

## Key Design Rules

1. **CLI is frozen.** `python app.py collect/train/recognize` and `./run.sh` must produce identical
   behavior before and after every phase. Run the regression test (Step 4) before merging anything.

2. **One engine, one instance.** `FaceRecognitionEngine` is instantiated once in `desktop_app.py`
   (or `web_server.py`) and passed by reference to every component. No second instance anywhere.

3. **Camera owned by one thread.** `CameraThread` is the only place `cv2.VideoCapture` is opened.
   The engine receives frames as numpy arrays — it never touches the camera directly.

4. **Engine is UI-agnostic.** `engine.py` must not import `PyQt6`, `cv2.imshow`, `print`, or `fastapi`.
   It is pure Python + numpy + torch + sklearn. This is what makes it reusable across surfaces.

5. **No silent failures.** Every engine method returns a structured dict with a success/error field.
   Callers (GUI, CLI, web) are responsible for displaying errors in their own way.

6. **Model portability.** The `model/` directory is the only artifact that needs to travel with the app.
   The AppImage can ship with a pre-bundled model or guide the user to train on first launch.

7. **AppImage, not .deb.** An AppImage is a single executable file with no install step, works on any
   Linux distro with FUSE support (Ubuntu 20.04+, Fedora, Arch, etc.), and can be run from a USB drive.
   This is the "use anywhere on Linux" goal made concrete.
