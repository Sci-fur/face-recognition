import os
import glob
import pickle
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from classifier import train as _classifier_train
from classifier import save_model as _save_model
from classifier import predict as _predict

_STAGES = [
    'Look straight ahead',
    'Slowly turn head LEFT',
    'Slowly turn head RIGHT',
    'Tilt head UP',
    'Tilt head DOWN',
    'Change expression (smile)',
    'Move CLOSER to camera',
    'Move FARTHER from camera',
    'Tilt head sideways (left/right)',
]
_STAGE_SIZE = max(1, 150 // len(_STAGES))


class FaceRecognitionEngine:
    def __init__(self, dataset_dir='dataset', model_dir='model'):
        self.dataset_dir = dataset_dirz
        self.model_dir = model_dir
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._mtcnn = None
        self._resnet = None
        self._pipeline = None
        self._classes = None
        self._centroids = None
        self._dist_thresholds = None
        self.model_loaded = False

    def _init_mtcnn(self):
        if self._mtcnn is None:
            self._mtcnn = MTCNN(
                keep_all=True,
                device=self._device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                post_process=True,
            )
        return self._mtcnn

    def _init_resnet(self):
        if self._resnet is None:
            self._resnet = InceptionResnetV1(
                pretrained='vggface2'
            ).eval().to(self._device)
        return self._resnet

    def load_model(self):
        svm_path = os.path.join(self.model_dir, 'face_svm.pkl')
        if not os.path.exists(svm_path):
            self.model_loaded = False
            self._pipeline = None
            self._classes = None
            self._centroids = None
            self._dist_thresholds = None
            return False
        from classifier import load_model
        self._pipeline, self._classes, self._centroids, self._dist_thresholds = load_model()
        self.model_loaded = self._pipeline is not None
        return self.model_loaded

    def get_dataset_info(self):
        if not os.path.isdir(self.dataset_dir):
            return {'persons': [], 'total_images': 0}
        persons = []
        total = 0
        for d in sorted(os.listdir(self.dataset_dir)):
            pdir = os.path.join(self.dataset_dir, d)
            if not os.path.isdir(pdir):
                continue
            count = len([
                f for f in os.listdir(pdir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            if count > 0:
                persons.append({'name': d, 'count': count})
                total += count
        return {'persons': persons, 'total_images': total}

    def collect_frame(self, frame, name):
        mtcnn = self._init_mtcnn()
        person_dir = os.path.join(self.dataset_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        existing = len([
            f for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        rgb = frame[:, :, ::-1].copy()
        boxes, probs = mtcnn.detect(rgb)
        result = {
            'saved': False,
            'count': existing,
            'face_bbox': None,
            'detection_confidence': 0.0,
            'stage_hint': self._get_stage_hint(existing),
            'complete': existing >= 150,
        }
        if boxes is not None and len(probs) > 0:
            best_idx = int(np.argmax(probs))
            prob = float(probs[best_idx])
            box = [int(round(v)) for v in boxes[best_idx]]
            x1, y1, x2, y2 = box
            result['face_bbox'] = (x1, y1, x2, y2)
            result['detection_confidence'] = prob
            if prob >= 0.95:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    fname = f'face_{existing:04d}.jpg'
                    fpath = os.path.join(person_dir, fname)
                    cv2.imwrite(fpath, face_crop)
                    result['saved'] = True
                    result['count'] = existing + 1
                    result['stage_hint'] = self._get_stage_hint(existing + 1)
                    result['complete'] = (existing + 1) >= 150
        return result

    def _get_stage_hint(self, count):
        idx = min(count // _STAGE_SIZE, len(_STAGES) - 1)
        return _STAGES[idx]

    def train(self, progress_callback=None):
        def _report(step, pct):
            if progress_callback:
                progress_callback(step, pct)

        if not os.path.isdir(self.dataset_dir):
            return {'success': False, 'error': 'dataset/ directory not found'}
        _report('Scanning dataset', 10)
        persons = sorted([
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
        ])
        if len(persons) < 2:
            return {'success': False, 'error': 'Need at least 2 persons'}
        all_paths = []
        all_labels = []
        for p in persons:
            pdir = os.path.join(self.dataset_dir, p)
            paths = sorted(glob.glob(os.path.join(pdir, '*.jpg')))
            all_paths.extend(paths)
            all_labels.extend([p] * len(paths))
        if not all_paths:
            return {'success': False, 'error': 'No images found'}

        # Load embedding cache (purge stale entries for deleted persons)
        cache_path = os.path.join(self.dataset_dir, '_embeddings.pkl')
        emb_cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                emb_cache = pickle.load(f)
            emb_cache = {k: v for k, v in emb_cache.items() if os.path.exists(k)}

        _report('Extracting embeddings', 20)
        mtcnn = self._init_mtcnn()
        resnet = self._init_resnet()
        batch_size = 32
        all_embs = []
        all_valid = []
        new_cache = {}

        for i in range(0, len(all_paths), batch_size):
            batch_paths = all_paths[i:i + batch_size]
            batch_labels = all_labels[i:i + batch_size]
            face_batch = []
            idx_batch = []
            for idx, path in enumerate(batch_paths):
                cached = emb_cache.get(path)
                if cached is not None:
                    all_embs.append(cached)
                    all_valid.append(batch_labels[idx])
                    continue
                img = Image.open(path).convert('RGB')
                img.thumbnail((320, 240))
                face = mtcnn(img)
                if face is None:
                    continue
                if face.dim() == 3:
                    face = face.unsqueeze(0)
                face_batch.append(face)
                idx_batch.append(idx)
            if face_batch:
                faces_tensor = torch.cat(face_batch, dim=0)
                with torch.no_grad():
                    embs = resnet(faces_tensor.to(self._device)).cpu().numpy()
                for k, emb in enumerate(embs):
                    if not np.any(np.isnan(emb)):
                        all_embs.append(emb)
                        all_valid.append(batch_labels[idx_batch[k]])
                        new_cache[batch_paths[idx_batch[k]]] = emb
            pct = 20 + int(60 * (min(i + batch_size, len(all_paths)) / len(all_paths)))
            _report('Extracting embeddings', pct)

        # Update cache with new embeddings
        if new_cache:
            emb_cache.update(new_cache)
            with open(cache_path, 'wb') as f:
                pickle.dump(emb_cache, f)

        if len(all_embs) < 2:
            return {'success': False, 'error': 'Not enough valid embeddings'}
        _report('Training SVM', 85)
        X = np.array(all_embs)
        y = np.array(all_valid)
        pipeline, (train_acc, test_acc), _, centroids, thresholds = _classifier_train(X, y)
        _report('Saving model', 95)
        classes = sorted(set(all_valid))
        _save_model(pipeline, classes, centroids, thresholds)
        self._pipeline = pipeline
        self._classes = np.array(classes)
        self._centroids = centroids
        self._dist_thresholds = thresholds
        self.model_loaded = True
        _report('Done', 100)
        return {
            'success': True,
            'train_acc': float(train_acc),
            'val_acc': float(test_acc),
            'n_persons': len(persons),
            'n_images': len(all_embs),
            'error': None,
        }

    def recognize_frame(self, frame):
        if not self.model_loaded:
            return []
        mtcnn = self._init_mtcnn()
        resnet = self._init_resnet()
        rgb = frame[:, :, ::-1].copy()
        boxes, probs = mtcnn.detect(rgb)
        aligned = mtcnn(rgb)
        if boxes is None or aligned is None:
            return []
        results = []
        n = min(len(boxes), aligned.size(0))
        for i in range(n):
            face_tensor = aligned[i].unsqueeze(0)
            emb = None
            with torch.no_grad():
                emb = resnet(face_tensor.to(self._device))[0].cpu().numpy()
            name, confidence = _predict(
                self._pipeline, self._classes, emb,
                self._centroids, self._dist_thresholds,
            )
            box = [int(round(v)) for v in boxes[i]]
            x1, y1, x2, y2 = box
            results.append({
                'name': name,
                'confidence': float(confidence),
                'is_unknown': name == 'Unknown',
                'bbox': (x1, y1, x2, y2),
            })
        return results

    def delete_person(self, name):
        import shutil
        pdir = os.path.join(self.dataset_dir, name)
        if not os.path.isdir(pdir):
            return False
        shutil.rmtree(pdir)
        return True

