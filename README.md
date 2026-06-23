# Face Recognition System

A real-time face recognition CLI application using OpenCV + facenet-pytorch (InceptionResnetV1) + SVM classification. Built as a **Computer Peripheral Lab** course project.

---

## Features

- **Real-time recognition** via laptop webcam / Mobile camera
- **Multi-face support** — detects and recognizes multiple people simultaneously
- **Guided data collection** — 9 pose/expression stages for diverse training data
  - **Deep learning embeddings** — 512-dimensional features from InceptionResnetV1 (pretrained on VGGFace2)

- **SVM classification** with confidence threshold (rejects unknowns below 60%)
- **Unknown face rejection** — displays "Unknown" for unrecognized faces

---

## Project Structure

```
FaceRecognition/
├── app.py              # CLI entry point
├── collect.py          # Guided data collection (9 stages, 150 images)
├── train.py            # Training pipeline (embeddings + SVM)
├── recognize.py        # Real-time multi-face recognition
├── detector.py         # Face detection + alignment (MTCNN)
├── embedder.py         # Face embedding (InceptionResnetV1)
├── classifier.py       # SVM classifier (train/save/load/predict)
├── SRS.md              # Software Requirements Specification
├── requirements.txt    # Python dependencies
├── run.sh              # Convenience launcher (uses venv)
├── dataset/            # Collected face images (auto-generated)
│   ├── Alice/
│   └── Bob/
└── model/              # Trained models (auto-generated)
    ├── face_svm.pkl
    ├── scaler.pkl
    └── classes.npy
```

---

## Setup

### Requirements

- Python 3.10+
- Webcam (built-in or USB)
- ~2 GB free disk for model weights + dependencies

### Install

```bash
# Navigate to project directory
cd FaceRecognition

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Or use the pre-configured venv:
```bash
./run.sh --help
```

---

## Usage

### 1. Collect face data

Run once per person. Follow the on-screen instructions — the system guides you through 9 different poses for diverse training data.

```bash
./run.sh collect --name "Saifur"
./run.sh collect --name "Kamrul"
```

During collection:
- Face must be clearly visible (green box)
- Each stage prompts a different pose/angle
- Progress bar and count shown on screen
- Press **Q** to quit early

### 2. Train the model

```bash
./run.sh train
```

Output:
```
Found 2 persons: ['Alice', 'Bob']
  Alice: 150 images
  Bob: 150 images
Total images: 300
Extracting face embeddings...
Training accuracy:   100.00%
Validation accuracy: 97.33%
Model saved to model/ with 2 classes: ['Alice', 'Bob']
```

### 3. Run real-time recognition

```bash
./run.sh recognize
```

- Green box = recognized person with name + confidence
- Red box = unknown person
- Press **Q** to quit

---

## How It Works

```
Webcam Frame
    │
    ▼
MTCNN Face Detection ───► Bounding boxes for all faces
    │
    ▼
Face Alignment (landmark-based)
    │
    ▼
InceptionResnetV1 ───► 512-dim embedding
    │
    ▼
SVM Classifier ───► Name + Confidence
    │
    ▼
Label Overlay on Video
```

- **Detection**: MTCNN (Multi-Task Cascaded CNN) — detects faces + landmarks
- **Embedding**: InceptionResnetV1 pretrained on VGGFace2 (3.3M images)
- **Classification**: SVM with RBF kernel + StandardScaler
- **Confidence threshold**: 60% — below this is "Unknown"

---

## Accuracy Tips

| Factor | Impact |
|--------|--------|
| Diverse poses during collection | High |
| Good lighting (not too dark/bright) | High |
| 100–150 images per person | Medium |
| Look directly at camera during recognition | Medium |
| Similar distance as during collection | Low |

---

## Data Flow

```
collect.py                train.py                  recognize.py
    │                        │                          │
    ▼                        ▼                          ▼
Webcam ──► dataset/ ──► MTCNN + ResNet ──► model/ ──► Webcam ──► Output
         Alice/        (face detection     SVM        (live feed)  labeled
         Bob/           + embedding)       .pkl                    frames
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0 | Deep learning framework |
| facenet-pytorch | ≥2.5 | MTCNN + InceptionResnetV1 |
| opencv-python | ≥4.7 | Camera capture + image processing |
| scikit-learn | ≥1.0 | SVM classifier |
| numpy | ≥1.24 | Numerical operations |
| Pillow | ≥10.0 | Image loading |
| joblib | ≥1.1 | Model serialization |
| tqdm | ≥4.64 | Progress bars |

---

## Limitations

- Accuracy drops in poor lighting or extreme angles
- Performance decreases with >4 faces in frame (CPU-bound)
- Unknown faces can be misclassified if too similar to known faces
- Model weights downloaded on first run (~310 MB total: MTCNN + VGGFace2)

---

## Course Context

**Course**: Computer Peripheral Lab  
**Peripheral**: Webcam (USB Video Class device)  
**Concepts demonstrated**:
- Video capture and processing as a peripheral I/O operation
- Real-time image processing pipeline
- Integration of camera hardware with software
- Human-computer interaction via visual output
# face-recognition
