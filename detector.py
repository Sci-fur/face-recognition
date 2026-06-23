import torch
from facenet_pytorch import MTCNN as _MTCNN
import numpy as np

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

_mtcnn = _MTCNN(
    keep_all=True,
    device=_device,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    post_process=True,
)


def detect_faces(image):
    rgb = image[:, :, ::-1].copy()
    boxes, probs = _mtcnn.detect(rgb)
    if boxes is None:
        return []
    results = []
    for box, prob in zip(boxes, probs):
        box = [int(round(v)) for v in box]
        x1, y1, x2, y2 = box
        results.append({
            'box': (x1, y1, x2, y2),
            'confidence': float(prob),
        })
    return results


def get_aligned_face(image):
    rgb = image[:, :, ::-1].copy()
    faces = _mtcnn(rgb)
    if faces is None:
        return None
    if faces.dim() == 4:
        return faces[0:1]
    return faces.unsqueeze(0)


def get_all_boxes_and_faces(image):
    rgb = image[:, :, ::-1].copy()
    boxes, probs = _mtcnn.detect(rgb)
    aligned = _mtcnn(rgb)
    if boxes is None or aligned is None:
        return []
    results = []
    n = min(len(boxes), aligned.size(0))
    for i in range(n):
        box = [int(round(v)) for v in boxes[i]]
        x1, y1, x2, y2 = box
        face_tensor = aligned[i].unsqueeze(0)
        results.append({
            'box': (x1, y1, x2, y2),
            'detection_confidence': float(probs[i]),
            'aligned': face_tensor,
        })
    return results


def get_aligned_faces_batch(image_paths):
    from PIL import Image
    faces_list = []
    valid_indices = []
    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert('RGB')
        face = _mtcnn(img)
        if face is not None:
            if face.dim() == 3:
                face = face.unsqueeze(0)
            faces_list.append(face)
            valid_indices.append(idx)
    if len(faces_list) == 0:
        return None, []
    faces = torch.cat(faces_list, dim=0)
    return faces, valid_indices
