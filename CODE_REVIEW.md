# Code Review Guide — Sequential Reading Order

Read files in this order to understand the project from high-level design → core engine → CLI frontend → GUI frontend.

---

## Phase 1 — Understand the Project

Start here to get context, requirements, and architecture decisions.

| # | File | Purpose | Est. time |
|---|---|---|---|
| 1 | `README.md` | Project overview, how to set up and run everything | 2 min |
| 2 | `SRS.md` | Software Requirements Specification (10 functional, 5 non-functional) | 5 min |
| 3 | `Plan.md` | Three-phase architecture plan, key decisions along the way | 5 min |

---

## Phase 2 — Core Engine (engine.py)

This is the heart of the project — all frontends (CLI, GUI) call this API.

| # | File | Purpose | Est. time |
|---|---|---|---|
| 4 | `engine.py` | `FaceRecognitionEngine` class — owns detector/embedder/classifier, exposes `load_model()`, `get_dataset_info()`, `collect_frame()`, `train()`, `recognize_frame()`, `delete_person()` | 8 min |

**Key insight:** Engine creates its own MTCNN and ResNet lazily (one-time init). It does NOT open a camera — it receives `numpy` arrays and returns results.

---

## Phase 3 — Individual Components

These are the building blocks that `engine.py` uses internally.

| # | File | Purpose | Est. time |
|---|---|---|---|
| 5 | `detector.py` | MTCNN face detection + alignment (module-level singleton) | 2 min |
| 6 | `embedder.py` | InceptionResnetV1 → 512-dim face embedding (module-level singleton) | 1 min |
| 7 | `classifier.py` | SVM (RBF) + StandardScaler + centroid-based distance rejection. Methods: `train()`, `save()`, `load()`, `predict()`. | 5 min |

**Data flow:** `frame → MTCNN → aligned face → ResNet → 512-vector → SVM → name + confidence`

---

## Phase 4 — CLI Frontend

Command-line interface using these scripts, coordinated by `app.py`.

| # | File | Purpose | Est. time |
|---|---|---|---|
| 8 | `app.py` | Argparse-based CLI entry point — dispatches to collect/train/recognize | 2 min |
| 9 | `collect.py` | Guided 9-stage data collection (~150 images/person with pose hints) | 3 min |
| 10 | `train.py` | Calls `engine.train()` with a progress callback, saves model to `model/` | 1 min |
| 11 | `recognize.py` | Real-time webcam recognition with 8-frame bounding-box persistence | 3 min |

---

## Phase 5 — GUI Frontend

PyQt6 desktop application. The GUI uses `CameraThread` (QThread) so the UI stays responsive during camera capture and model inference.

| # | File | Purpose | Est. time |
|---|---|---|---|
| 12 | `gui/utils.py` | `numpy_to_pixmap()` and `draw_boxes()` — shared GUI helpers | 1 min |
| 13 | `gui/camera_thread.py` | `CameraThread(QThread)` — captures frames, runs recognition/collection, emits signals | 3 min |
| 14 | `gui/collect_dialog.py` | `CollectDialog` — modal dialog for adding a new person (name input, progress bar, stage hint) | 3 min |
| 15 | `gui/train_dialog.py` | `TrainDialog` + `TrainWorker(QThread)` — modal dialog for retraining the model | 3 min |
| 16 | `gui/main_window.py` | `MainWindow` — main application window (sidebar with mode radio buttons, person list, video QLabel, log panel, FPS counter) | 8 min |
| 17 | `desktop_app.py` | QApplication entry point — sets `QT_PLUGIN_PATH`, creates `MainWindow`, shows it | 1 min |

---

## Phase 6 — Launchers and Config

| # | File | Purpose | Est. time |
|---|---|---|---|
| 18 | `run.sh` | CLI launcher — `venv/bin/python app.py "$@"` | 30 s |
| 19 | `run_desktop.sh` | GUI launcher — resolves symlink, `exec venv/bin/python desktop_app.py` | 30 s |
| 20 | `requirements.txt` | Pip dependency list | 30 s |
| 21 | `session.md` | Development session log (what was done and when) | 2 min |

---

## Architecture Diagram (Data Flow)

```
                         ┌─────────────────────┐
                         │   Webcam (cv2)       │
                         │  640×480 BGR frame   │
                         └────────┬────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │        FaceRecognitionEngine              │
          │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
          │  │ detector  │  │ embedder │  │classif │ │
          │  │ (MTCNN)   │→ │ (ResNet) │→ │ (SVM)  │ │
          │  └──────────┘  └──────────┘  └────────┘ │
          └──────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            ┌──────────────┐          ┌──────────────────┐
            │  CLI Frontend│          │  GUI Frontend     │
            │  (app.py)    │          │  (desktop_app.py) │
            │  Terminal    │          │  PyQt6 Window     │
            └──────────────┘          └──────────────────┘
```

---

## Key Patterns

- **Singleton components:** `detector.py` / `embedder.py` keep module-level singletons (legacy). `engine.py` creates its own fresh instances for isolation.
- **Frame skip + persistence:** Both CLI and GUI skip every other frame for performance but persist bounding boxes for 8 frames to prevent flickering.
- **Unknown rejection:** Two-layer check in `classifier.py` — centroid distance (mean + 2.5σ) catches unknowns before SVM probability threshold (60%).
- **One engine, many surfaces:** All UI code imports `engine.py` — no business logic lives in UI files.
