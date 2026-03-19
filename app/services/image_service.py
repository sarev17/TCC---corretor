import cv2
import numpy as np

def detect_regions(pil_image):
    img = np.array(pil_image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binarização
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # encontrar contornos
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # filtrar ruídos pequenos
        if w > 50 and h > 50:
            boxes.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            })

    return boxes