import os
import cv2
from engine import FaceRecognitionEngine

_CAMERA = int(os.environ.get('CAMERA_INDEX', '0'))
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = FaceRecognitionEngine()
    return _engine


def collect(name):
    engine = _get_engine()
    cap = cv2.VideoCapture(_CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print('Error: Could not open webcam.')
        return
    print(f'Collecting faces for "{name}". Press Q to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        result = engine.collect_frame(frame, name)
        total = result['count']
        if result['face_bbox']:
            x1, y1, x2, y2 = result['face_bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
        bar_w = w - 40
        fill = int(bar_w * min(total / 150, 1.0))
        cv2.rectangle(display, (20, h - 90), (20 + bar_w, h - 80), (80, 80, 80), 1)
        cv2.rectangle(display, (20, h - 90), (20 + fill, h - 80), (0, 255, 0), -1)
        cv2.putText(display, f'{total}/150', (25, h - 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, result['stage_hint'], (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        if not result['face_bbox']:
            cv2.putText(display, 'No face detected', (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(display, 'Face detected', (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Face Collection', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if result['complete']:
            print(f'\nReached 150 images for "{name}"')
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f'Done! Collected {total} images for "{name}"')
