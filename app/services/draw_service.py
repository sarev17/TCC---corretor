import cv2
import numpy as np

def draw_boxes(pil_image, boxes):
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
            (0, 0, 255),  # vermelho
            2
        )

    return img