import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def numpy_to_pixmap(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


def draw_boxes(frame, results):
    out = frame.copy()
    for r in results:
        x1, y1, x2, y2 = r['bbox']
        is_unknown = r.get('is_unknown', False)
        name = r.get('name', 'Unknown')
        confidence = r.get('confidence', 0.0)
        color = (0, 0, 255) if is_unknown else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{name} ({confidence:.0%})'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_label = max(y1 - th - 10, 0)
        cv2.rectangle(out, (x1, y1_label), (x1 + tw + 10, y1), color, -1)
        cv2.putText(out, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out
