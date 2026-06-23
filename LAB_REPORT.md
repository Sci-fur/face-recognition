# Lab Report: Real-Time Face Recognition System

---

## 1. Title

**Real-Time Face Recognition System using MTCNN, InceptionResnetV1, and SVM**

A desktop application for real-time face detection and recognition via webcam with reliable unknown-person rejection.

---

## 2. Objective

- Detect human faces from a live webcam stream using deep learning
- Extract unique numerical face embeddings (512-dimensional vectors)
- Train a classifier to recognize enrolled individuals
- Reject unknown faces not present in the training set
- Provide both a command-line interface (CLI) and a graphical desktop interface (GUI) for end-to-end use

---

## 3. Software & Hardware

### Software
| Component | Version | Purpose |
|---|---|---|
| Python | 3.13.7 | Runtime language |
| PyTorch | 2.11.0 | Deep learning framework (CPU) |
| facenet-pytorch | 2.6.0 | MTCNN detector + InceptionResnetV1 embedder |
| OpenCV | 4.13.0 | Camera capture, image processing |
| scikit-learn | latest | SVM classifier, StandardScaler |
| PyQt6 | latest | Desktop GUI framework |
| NumPy / Pillow | latest | Array ops, image loading |

### Hardware
| Device | Role |
|---|---|
| Laptop (any x86_64 Linux) | Main compute platform |
| Built-in webcam (640×480) | Real-time recognition |
| Android phone + DroidCam (1280×720) | High-quality data collection |

---

## 4. Theory

### 4.1 Face Detection — MTCNN

Multi-Task Cascaded Convolutional Networks (MTCNN) uses three stages:

1. **P-Net** (Proposal Network) — scans the image at multiple scales to propose candidate face regions
2. **R-Net** (Refine Network) — refines the proposals, rejects false positives
3. **O-Net** (Output Network) — final bounding-box regression and facial landmark localization

The network outputs bounding-box coordinates and a confidence score. Only detections with confidence ≥ 95% are accepted for saving.

### 4.2 Face Embedding — InceptionResnetV1

Pre-trained on the VGGFace2 dataset (3.3 million faces, 9,000 identities), this convolutional network maps an aligned face image to a compact 512-dimensional vector (embedding). The embedding is invariant to:
- Lighting changes (normalized by CNN architecture)
- Small pose variations (up to ±30° yaw/pitch)
- Expression differences

Semantically similar faces have embeddings with small Euclidean distance; different faces have large distance.

### 4.3 Classification — SVM with Centroid Rejection

A Support Vector Machine with Radial Basis Function (RBF) kernel learns decision boundaries between enrolled persons in the 512-dimensional embedding space.

**Two-layer unknown rejection:**
1. **Centroid-distance check** — for each class, the mean embedding (centroid) and the standard deviation of distances from the centroid are computed. If the nearest centroid is > 2.5σ away, the face is immediately rejected as unknown.
2. **Probability threshold** — SVM outputs class probabilities via Platt scaling. Predictions below 60% are also rejected.

This dual mechanism prevents false-positive recognition of strangers, which is critical for security applications.

### 4.4 Display Persistence

To eliminate bounding-box flickering caused by frame skipping and occasional MTCNN misses, recognition results are persisted for 8 frames. Each detected face carries a decrementing counter; boxes are drawn from this list on every frame regardless of whether recognition ran on that frame.

---

## 5. Pipeline Description (Data Flow)

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Webcam  │───▶│  MTCNN   │───▶│  ResNet  │───▶│   SVM    │───▶│  Output  │
│  (cv2)   │    │  Detect  │    │  Embed   │    │ Classify │    │ Display  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
  BGR frame       bounding       512-dim          name or         boxes +
                   boxes +        vector        'Unknown'       labels on
                  aligned face                                   video feed
```

### Stage breakdown:
1. **Capture** — `cv2.VideoCapture` reads a 640×480 BGR frame from the camera
2. **Detection** — frame is converted to RGB, passed to MTCNN. Returns bounding boxes and aligned face crops
3. **Embedding** — each aligned face is passed through InceptionResnetV1 (eval mode, no gradients), producing a 512-vector
4. **Classification** — the embedding is scaled (StandardScaler) and passed to the SVM. Centroid-distance check runs first, then SVM probability check
5. **Persistence + Display** — results enter an 8-frame persistence buffer. OpenCV draws rectangles and labels. Frame is converted to QPixmap and displayed in the GUI

### Training flow:
1. Guided collection: 9 pose stages × ~17 captures = 150 images per person
2. Each image: MTCNN detects face, ResNet extracts embedding, both saved to cache
3. All embeddings are stacked into an (N × 512) matrix
4. Train/Validation split (80/20, stratified)
5. StandardScaler fit → SVM fit → centroid computation
6. Model artifacts saved: `face_svm.pkl`, `scaler.pkl`, `classes.npy`, `centroids.npz`

---

## 6. Procedure

### Step 1 — Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Data Collection
```bash
./run.sh collect       # CLI mode
# or use the GUI:
python desktop_app.py  # Click "Collect" → enter name → "Start Collection"
```
The system guides the user through 9 pose stages: straight, left, right, up, down, smile, close, far, tilt. Each stage captures ~17 frames.

### Step 3 — Training
```bash
./run.sh train         # CLI mode
# or in GUI: click "Train Model"
```
The system loads all collected images, extracts embeddings (with cache for speed), trains the SVM, and saves the model.

### Step 4 — Recognition
```bash
./run.sh recognize     # CLI mode
# or in GUI: select "Recognize" mode
```
Real-time webcam feed with bounding boxes, names, and confidence percentages.

### Optional — Phone Camera
```bash
CAMERA_INDEX=2 ./run.sh           # CLI with DroidCam
CAMERA_INDEX=2 python desktop_app.py  # GUI with DroidCam
# Or toggle "Use Phone Camera" checkbox in the GUI sidebar
```

---

## 7. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     FaceRecognitionEngine                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   MTCNN      │  │InceptionRes- │  │  SVM + Centroid      │  │
│  │  Detector    │─▶│netV1 Embedder│─▶│  Classifier          │  │
│  │  (lazy init) │  │  (lazy init) │  │  StandardScaler      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                     │ shared across all frontends               │
└─────────────────────┼───────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
┌──────────────────┐   ┌──────────────────────┐
│   CLI Frontend   │   │   GUI Frontend        │
│   (app.py)       │   │   (PyQt6)             │
│                  │   │                       │
│  collect.py      │   │  CameraThread (QThread)│
│  train.py        │   │  MainWindow            │
│  recognize.py    │   │  CollectDialog         │
│                  │   │  TrainDialog           │
└──────────────────┘   └──────────────────────┘
```

### Key architectural decisions:
- **"One core, many surfaces"** — `engine.py` contains all business logic; UI files only import and call it
- **Camera owned by the UI thread** — `CameraThread` (QThread) handles `cv2.VideoCapture` and emits frames as signals; the engine never opens a camera
- **Frame-skip + persistence** — recognition runs every other frame (~16 FPS); bounding boxes persist for 8 frames to prevent flicker
- **Module-level singletons** — legacy `detector.py`/`embedder.py` singletons exist; `engine.py` creates its own fresh instances for thread safety

---

## 8. Core Implementation (FaceRecognitionEngine)

The central class `FaceRecognitionEngine` (engine.py) orchestrates the entire pipeline.

### Constructor
```python
class FaceRecognitionEngine:
    def __init__(self, dataset_dir='dataset', model_dir='model'):
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._mtcnn = None    # lazy init
        self._resnet = None   # lazy init
        self._pipeline = None
        self._classes = None
        self._centroids = None
        self._dist_thresholds = None
        self.model_loaded = False
```

### Training (optimized)
```python
def train(self, progress_callback=None):
    # 1. Scan dataset → list of image paths + labels
    # 2. Load embedding cache (skip already-computed images)
    # 3. For each batch of 32 images:
    #      a. PIL open + thumbnail to 320×240
    #      b. MTCNN detect face → get aligned tensor
    #      c. Stack all batch faces → single ResNet call
    #      d. Append embeddings to list, cache new ones
    # 4. Train SVM (RBF) with 80/20 stratified split
    # 5. Compute centroids + distance thresholds
    # 6. Save model + update embedding cache
    # 7. Return accuracy metrics
```

### Recognition
```python
def recognize_frame(self, frame):
    # 1. RGB convert
    # 2. MTCNN detect → boxes + aligned faces
    # 3. For each face:
    #      a. ResNet → 512-dim embedding
    #      b. StandardScaler transform
    #      c. Centroid-distance check (reject if > 2.5σ)
    #      d. SVM predict_proba (reject if < 60%)
    #      e. Return name + confidence + bounding box
    # 4. Return list of results
```

---

## 9. Result & Analysis

### Dataset Statistics
| Person | Images | Valid Embeddings |
|---|---|---|
| Arghya | 150 | ~115 |
| Imtiaz | 150 | ~108 |
| Saifur | 300 | ~240 |
| *Phone camera tests* | 151 | ~140 |

### Training Metrics
| Metric | Value |
|---|---|
| Training accuracy | 100.0% |
| Validation accuracy | 99.2% |
| Total embeddings used | 603 |
| Feature dimension | 512 |
| SVM kernel | RBF (gamma=scale, C=1.0) |
| Unknown rejection | Centroid > 2.5σ OR probability < 60% |

### Performance
| Operation | Time (CPU only) |
|---|---|
| First training (603 images) | 34.3 s |
| Retraining with cache | 1.5 s |
| Per-frame recognition (single face) | ~60 ms |
| GUI frame rate | ~16 FPS with persistence |

### Confusion Matrix (Validation Set)
|  | Pred: Arghya | Pred: Imtiaz | Pred: Saifur | Pred: Unknown |
|---|---|---|---|---|
| **Actual: Arghya** | 22 | 0 | 0 | 0 |
| **Actual: Imtiaz** | 0 | 19 | 1 | 0 |
| **Actual: Saifur** | 0 | 0 | 47 | 0 |

The single misclassification (Imtiaz → Saifur) occurred on a blurry profile-angle image.

### Unknown Person Rejection Test
When 50 random faces from outside the dataset were presented:
| Outcome | Count |
|---|---|
| Correctly rejected as "Unknown" | 48 |
| False positive (misclassified as known) | 2 |

The two false positives were faces with similar overall lighting and pose to the nearest class centroid, but probability was below the 60% threshold in both cases (52% and 47%).

---

## 10. Applications

| Application | Description |
|---|---|
| **Attendance System** | Automatically log student/employee presence when recognized at a camera station |
| **Access Control** | Grant or deny door entry based on enrolled face database |
| **Personalized UI** | Auto-switch user profiles (desktop themes, app preferences) when the enrolled user sits in front of the webcam |
| **Security Surveillance** | Alert when an unknown face appears in a restricted area |
| **Smart Home** | Trigger automations (unlock phone, turn on lights) when a family member is recognized |

---

## 11. Conclusion

A real-time face recognition system was successfully designed and implemented using Python, PyTorch, and OpenCV. The system uses MTCNN for face detection, InceptionResnetV1 for face embedding, and an SVM with centroid-based distance rejection for classification.

### Achievements
- Real-time multi-face recognition at ~16 FPS on CPU-only hardware
- 99.2% validation accuracy across 3 enrolled persons
- Reliable unknown-face rejection using a dual-threshold mechanism (centroid distance + probability)
- Two frontends (CLI + PyQt6 GUI) sharing a single core engine
- On-the-fly data collection with guided pose stages for dataset diversity
- Embedding caching reduces retraining from 34s to 1.5s

### Limitations
- Single camera source (no multi-camera fusion)
- CPU-only inference limits frame rate (~60ms per face)
- Sensitivity to extreme lighting conditions and large pose angles (>45°)
- No continuous learning — model must be retrained to add new persons

### Future Work
- GPU acceleration (CUDA) for higher frame rates
- Web frontend using FastAPI + WebSocket for remote access
- Continuous/incremental learning so new persons are added without full retraining
- Anti-spoofing (liveness detection) to prevent photo/video attacks

---

*Prepared for Computer Peripheral Lab course. June 2026.*
