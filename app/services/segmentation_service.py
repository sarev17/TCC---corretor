"""
Serviço responsável por segmentar questões da prova.

Recursos:
- OCR para detectar "Questão"
- suporte a 2 colunas
- merge inteligente entre colunas (resolve questão quebrada)
"""

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

        # tolerância OCR
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
# 🚀 MERGE INTELIGENTE ENTRE COLUNAS
# =========================
def _merge_cross_column_questions(regions: List[Dict], height: int) -> List[Dict]:

    left = [r for r in regions if r["x"] == 0]
    right = [r for r in regions if r["x"] != 0]

    merged = []
    used_right = set()

    for l in left:
        l_bottom = l["y"] + l["h"]

        # se encosta no final → provável continuação
        if l_bottom > height * 0.85:

            for i, r in enumerate(right):
                if i in used_right:
                    continue

                # começa no topo → continuação
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

    # adiciona regiões da direita não usadas
    for i, r in enumerate(right):
        if i not in used_right:
            merged.append(r)

    return sorted(merged, key=lambda r: (r["x"], r["y"]))


# =========================
# FUNÇÃO PRINCIPAL
# =========================
def segment_questions(pil_image) -> List[Dict]:

    img = np.array(pil_image)
    height, width = img.shape[:2]

    mid = width // 2

    # dividir colunas
    left_img = img[:, :mid]
    right_img = img[:, mid:]

    # detectar posições
    left_positions = _detect_question_positions(left_img)
    right_positions = _detect_question_positions(right_img)

    # construir regiões
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

    # CORREÇÃO DO PROBLEMA DA QUESTÃO 3
    regions = _merge_cross_column_questions(regions, height)

    return regions