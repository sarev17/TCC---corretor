import cv2
import numpy as np
import pytesseract
from typing import List, Dict


# =========================
# PREPROCESSAMENTO
# =========================
def _preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

    return thresh


# =========================
# DETECTAR POSIÇÕES DE QUESTÕES
# =========================
def _detect_question_positions(image: np.ndarray) -> List[int]:
    processed = _preprocess(image)

    data = pytesseract.image_to_data(
        processed,
        lang="por",
        output_type=pytesseract.Output.DICT,
        config="--psm 6"
    )

    positions = []

    for i, text in enumerate(data["text"]):
        if not text:
            continue

        normalized = text.strip().lower()

        if normalized.startswith("quest"):
            y = data["top"][i]
            positions.append(y)

    return sorted(positions)


# =========================
# CONSTRUIR REGIÕES
# =========================
def _build_regions(positions: List[int], x_offset: int, width: int, height: int) -> List[Dict]:
    regions = []

    if not positions:
        return regions

    for i, y_start in enumerate(positions):

        if i < len(positions) - 1:
            y_end = positions[i + 1]
        else:
            y_end = height

        regions.append({
            "x": int(x_offset),
            "y": int(y_start),
            "w": int(width),
            "h": int(y_end - y_start)
        })

    return regions


# =========================
# MERGE ENTRE COLUNAS
# =========================
def _merge_cross_column_questions(regions: List[Dict], height: int) -> List[Dict]:

    left = [r for r in regions if r["x"] == 0]
    right = [r for r in regions if r["x"] != 0]

    merged = []
    used_right = set()

    for l in left:
        l_bottom = l["y"] + l["h"]

        if l_bottom > height * 0.85:
            for i, r in enumerate(right):
                if i in used_right:
                    continue

                if r["y"] < height * 0.25:
                    merged.append({
                        "x": l["x"],
                        "y": l["y"],
                        "w": l["w"] + r["w"],
                        "h": (height - l["y"])
                    })

                    used_right.add(i)
                    break
            else:
                merged.append(l)
        else:
            merged.append(l)

    for i, r in enumerate(right):
        if i not in used_right:
            merged.append(r)

    return merged


# =========================
# 🔥 ORDEM DE LEITURA CORRETA
# =========================
def _sort_reading_order(regions: List[Dict], mid: int) -> List[Dict]:
    left = []
    right = []

    for r in regions:
        if r["x"] < mid:
            left.append(r)
        else:
            right.append(r)

    left = sorted(left, key=lambda r: r["y"])
    right = sorted(right, key=lambda r: r["y"])

    return left + right


# =========================
# FUNÇÃO PRINCIPAL
# =========================
def segment_questions(pil_image) -> List[Dict]:

    img = np.array(pil_image)
    height, width = img.shape[:2]

    mid = width // 2

    left_img = img[:, :mid]
    right_img = img[:, mid:]

    left_positions = _detect_question_positions(left_img)
    right_positions = _detect_question_positions(right_img)

    left_regions = _build_regions(
        left_positions,
        x_offset=0,
        width=mid,
        height=height
    )

    right_regions = _build_regions(
        right_positions,
        x_offset=mid,
        width=mid,
        height=height
    )

    regions = left_regions + right_regions

    regions = _merge_cross_column_questions(regions, height)

    # 🔥 AQUI É O MAIS IMPORTANTE
    regions = _sort_reading_order(regions, mid)

    return regions