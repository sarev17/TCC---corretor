"""
Serviço para desenhar bounding boxes (debug visual).
"""

import cv2
import numpy as np
from typing import List, Dict


def draw_boxes(pil_image, boxes: List[Dict], color=(0, 255, 0)):
    """
    Desenha caixas na imagem.

    :param pil_image: imagem PIL ou numpy
    :param boxes: lista de regiões
    :param color: cor (BGR)
    :return: imagem com boxes
    """

    if isinstance(pil_image, np.ndarray):
        img = pil_image.copy()
    else:
        img = np.array(pil_image)

    for box in boxes:
        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

    return img