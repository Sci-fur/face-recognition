# Session Summary

## Project: Face Recognition System (Computer Peripheral Lab)

---

## What was built

A **real-time face recognition CLI app** using Python + OpenCV + facenet-pytorch (InceptionResnetV1) + SVM. Uses a laptop webcam as the peripheral device to detect and recognize multiple faces simultaneously.

## Files created (11)

| File | Purpose |
|---|---|
| `SRS.md` | Comprehensive Software Requirements Specification (7 sections, 10 FR, 5 NFR) |
| `app.py` | CLI entry point (`collect`, `train`, `recognize`) |
| `detector.py` | MTCNN face detection + alignment |
| `embedder.py` | InceptionResnetV1 (VGGFace2) → 512-dim embeddings |
| `classifier.py` | SVM (RBF) + centroid-based distance rejection |
| `collect.py` | Guided data collection (9 pose stages, 150 images) |
| `train.py` | Training pipeline (embeddings → SVM → centroids) |
| `recognize.py` | Real-time multi-face recognition with persistence |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `run.sh` | Convenience launcher using venv |

---

## Issues encountered & fixes

### 1. MTCNN batch dimension error
```
Exception: MTCNN batch processing only compatible with equal-dimension images
```
**Fix:** Changed `get_aligned_faces_batch()` to process images one-by-one, returning only valid detections.

### 2. Embedding dimension mismatch (512 vs 1024)
```
X has 1024 features, but StandardScaler is expecting 512 features as input.
```
**Fix:** `get_aligned_face()` with `keep_all=True` returned ALL faces. When 2 faces were detected, MTCNN returned both → `get_embedding` flattened both (2×512=1024). Fixed by taking only `[0:1]` face and using `[0]` indexing in `get_embedding`.

### 3. Flickering bounding box
Box/label blinked on/off every 2nd frame (frame skipping) and on detection failure.

**Fix:** Added per-face persistence (`_PERSIST_FRAMES = 8`) — keeps displaying last known detection for 8 frames after it's lost.

### 4. Unknown faces misclassified as known
SVM with only 2 classes was overconfident — mapped any face-like embedding to "saifur" or "Linus" with high probability.

**Fix:** Added **centroid-based distance rejection**:
   - During training: compute mean embedding (centroid) + distance threshold (`mean + 2.5σ`) per class
   - During prediction: if test embedding falls outside all class boundaries, return "Unknown"
   - Confidence is proportionally reduced near boundaries

### 5. Poor training data diversity
100 near-identical frames from webcam don't generalize well.

**Fix:** Rewrote `collect.py` with 9 guided pose stages (straight, left, right, up, down, expressions, close, far, tilt) totaling 150 diverse images per person.

---

## Tech stack

| Component | Choice |
|---|---|
| Language | Python 3.13 |
| Face detection | MTCNN (facenet-pytorch) |
| Deep learning | InceptionResnetV1 pre-trained on VGGFace2 (512-dim) |
| Classifier | SVM (RBF kernel) + StandardScaler |
| Rejection | Centroid-distance check (`mean + 2.5σ`) + probability threshold (60%) |
| Computer vision | OpenCV 4.13 |
| CLI | argparse |

---

## How to use

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or use the wrapper
./run.sh --help

# Collect diverse training data
./run.sh collect --name "Alice"

# Train the model
./run.sh train

# Recognize faces in real-time
./run.sh recognize
```

---

## Performance

- **Training accuracy:** 100% (overfits on small data, mitigated by centroid rejection)
- **Validation accuracy:** 100%
- **Real-time FPS:** ~5–10 FPS on CPU (MTCNN + InceptionResnetV1)
- **Unknown rejection:** Distance-based (class boundary) + probability-based (60% threshold)
