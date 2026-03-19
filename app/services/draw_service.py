"""
Serviço para desenhar bounding boxes (debug visual).
"""

import cv2
import numpy as np
from typing import List, Dict

def draw_boxes(pil_image, boxes: List[Dict], color=(0, 255, 0)):
    if isinstance(pil_image, np.ndarray):
        img = pil_image.copy()
    else:
        img = np.array(pil_image)

    for box in boxes:
        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        c = box.get("color", color)

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            c,
            2
        )

    return img