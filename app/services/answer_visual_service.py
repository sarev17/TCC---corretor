import cv2
import numpy as np
import os
from typing import List, Dict


DEBUG_DIR = "debug_questions"


def _ensure_debug_dir(q_index):
    path = f"{DEBUG_DIR}/q{q_index}"
    os.makedirs(path, exist_ok=True)
    return path


def _preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def _extract_line_candidates(question_img, debug_path=None):
    gray = _preprocess(question_img)

    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    if debug_path:
        cv2.imwrite(f"{debug_path}/1_gray.png", gray)
        cv2.imwrite(f"{debug_path}/2_threshold.png", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    if debug_path:
        cv2.imwrite(f"{debug_path}/3_dilated.png", dilated)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = question_img.shape[:2]
    candidates = []

    contour_img = question_img.copy()

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)

        # filtro mínimo só para remover ruído muito pequeno
        if w > w_img * 0.15 and h > 12:
            candidates.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if debug_path:
        cv2.imwrite(f"{debug_path}/4_contours_filtered.png", contour_img)

    candidates = sorted(candidates, key=lambda item: item["y"])
    return candidates


def _group_by_vertical_spacing(lines, gap_factor=1.8):
    """
    Agrupa linhas em blocos com base no espaçamento vertical.
    gap_factor define o quão maior o gap precisa ser para virar quebra de bloco.
    """
    if not lines:
        return []

    if len(lines) == 1:
        return [lines]

    gaps = []
    for i in range(len(lines) - 1):
        current_bottom = lines[i]["y"] + lines[i]["h"]
        next_top = lines[i + 1]["y"]
        gap = max(0, next_top - current_bottom)
        gaps.append(gap)

    positive_gaps = [g for g in gaps if g > 0]
    base_gap = np.median(positive_gaps) if positive_gaps else 0

    if base_gap <= 0:
        base_gap = 10

    groups = []
    current_group = [lines[0]]

    for i in range(1, len(lines)):
        prev = lines[i - 1]
        curr = lines[i]

        gap = curr["y"] - (prev["y"] + prev["h"])

        if gap > base_gap * gap_factor:
            groups.append(current_group)
            current_group = [curr]
        else:
            current_group.append(curr)

    groups.append(current_group)
    return groups


def _select_last_relevant_group(groups):
    """
    Por heurística, o último grupo é o bloco das alternativas.
    """
    if not groups:
        return []

    return groups[-1]


def detect_option_lines(question_img, q_index=0, debug=True) -> List[Dict]:
    debug_path = _ensure_debug_dir(q_index)

    if debug:
        cv2.imwrite(f"{debug_path}/0_original.png", question_img)

    candidates = _extract_line_candidates(
        question_img,
        debug_path=debug_path if debug else None
    )

    groups = _group_by_vertical_spacing(candidates, gap_factor=1.8)
    option_lines = _select_last_relevant_group(groups)

    if debug:
        debug_img = question_img.copy()

        # desenha todos os candidatos em amarelo
        for item in candidates:
            x, y, w, h = item["x"], item["y"], item["w"], item["h"]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 1)

        # desenha grupo final em verde
        for item in option_lines:
            x, y, w, h = item["x"], item["y"], item["w"], item["h"]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite(f"{debug_path}/5_lines_filtered.png", debug_img)

        for idx, line in enumerate(option_lines):
            x, y, w, h = line["x"], line["y"], line["w"], line["h"]
            roi = question_img[y:y+h, x:x+w]
            cv2.imwrite(f"{debug_path}/final_option_{idx}.png", roi)

    print(f"[Q{q_index}] grupos encontrados: {len(groups)}")
    print(f"[Q{q_index}] linhas finais: {len(option_lines)}")

    return option_lines