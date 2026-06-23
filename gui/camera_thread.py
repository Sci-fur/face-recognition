import cv2
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from gui.utils import numpy_to_pixmap, draw_boxes


class CameraThread(QThread):
    frame_ready = pyqtSignal(object)
    result_ready = pyqtSignal(list)
    collect_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    _FRAME_SKIP = 2
    _PERSIST_FRAMES = 8

    def __init__(self, engine, mode_ref, collect_name_ref, camera_index=0):
        super().__init__()
        self.engine = engine
        self.mode_ref = mode_ref
        self.collect_name_ref = collect_name_ref
        self._camera_index = camera_index
        self._running = False
        self._tracked = []

    def run(self):
        cap = cv2.VideoCapture(self._camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            self.error.emit('Could not open webcam.')
            return
        self._running = True
        frame_count = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.error.emit('Failed to read frame.')
                break
            frame_count += 1
            mode = self.mode_ref[0] if self.mode_ref else 'recognize'

            if mode == 'recognize':
                if frame_count % self._FRAME_SKIP == 0:
                    results = self.engine.recognize_frame(frame)
                    self._tracked = []
                    for r in results:
                        self._tracked.append({
                            'bbox': r['bbox'],
                            'name': r['name'],
                            'confidence': r['confidence'],
                            'is_unknown': r['is_unknown'],
                            'persist': self._PERSIST_FRAMES,
                        })
                    self.result_ready.emit(results)
                annotated = draw_boxes(frame, self._tracked)
                pix = numpy_to_pixmap(annotated)
                self.frame_ready.emit(pix)
                for t in self._tracked:
                    t['persist'] -= 1
                self._tracked = [t for t in self._tracked if t['persist'] > 0]

            elif mode == 'collect':
                name = self.collect_name_ref[0] if self.collect_name_ref else ''
                if name:
                    result = self.engine.collect_frame(frame, name)
                    display = frame.copy()
                    if result['face_bbox']:
                        x1, y1, x2, y2 = result['face_bbox']
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    pix = numpy_to_pixmap(display)
                    self.frame_ready.emit(pix)
                    self.collect_ready.emit(result)
                else:
                    pix = numpy_to_pixmap(frame)
                    self.frame_ready.emit(pix)

            else:
                pix = numpy_to_pixmap(frame)
                self.frame_ready.emit(pix)

            self.msleep(30)
        cap.release()

    def stop(self):
        self._running = False
        self.wait(2000)
