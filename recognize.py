import os
import cv2
import time
from engine import FaceRecognitionEngine

_CAMERA = int(os.environ.get('CAMERA_INDEX', '0'))
_engine = None
_FRAME_SKIP = 2
_PERSIST_FRAMES = 8


def _get_engine():
    global _engine
    if _engine is None:
        _engine = FaceRecognitionEngine()
    return _engine


def recognize():
    engine = _get_engine()
    loaded = engine.load_model()
    if not loaded:
        print('Error: No trained model found. Run "python app.py train" first.')
        return
    info = engine.get_dataset_info()
    names = [p['name'] for p in info['persons']]
    print(f'Loaded model with {len(names)} classes: {names}')
    print('Starting real-time recognition. Press Q to quit.')
    cap = cv2.VideoCapture(_CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print('Error: Could not open webcam.')
        return
    frame_count = 0
    prev_time = time.time()
    tracked = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        fps = 1.0 / (time.time() - prev_time + 1e-6)
        prev_time = time.time()
        frame_count += 1
        if frame_count % _FRAME_SKIP == 0:
            results = engine.recognize_frame(frame)
            tracked = []
            for r in results:
                tracked.append({
                    'bbox': r['bbox'],
                    'name': r['name'],
                    'confidence': r['confidence'],
                    'is_unknown': r['is_unknown'],
                    'persist': _PERSIST_FRAMES,
                })
        for t in tracked:
            if t['persist'] > 0:
                t['persist'] -= 1
                x1, y1, x2, y2 = t['bbox']
                color = (0, 0, 255) if t['is_unknown'] else (0, 255, 0)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label = f"{t['name']} ({t['confidence']:.0%})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y1_label = max(y1 - th - 10, 0)
                cv2.rectangle(display, (x1, y1_label), (x1 + tw + 10, y1), color, -1)
                cv2.putText(display, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tracked = [t for t in tracked if t['persist'] > 0]
        cv2.putText(display, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f'Faces: {len(tracked)}', (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Face Recognition', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
