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


# =========================
# DETECTAR LINHAS
# =========================
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

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

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

    return sorted(candidates, key=lambda item: item["y"])


# =========================
# AGRUPAMENTO
# =========================
def _group_by_vertical_spacing(lines, gap_factor=1.8):
    if not lines:
        return []

    if len(lines) == 1:
        return [lines]

    gaps = []
    for i in range(len(lines) - 1):
        bottom = lines[i]["y"] + lines[i]["h"]
        top = lines[i + 1]["y"]
        gaps.append(max(0, top - bottom))

    base_gap = np.median([g for g in gaps if g > 0]) if gaps else 10
    if base_gap <= 0:
        base_gap = 10

    groups = []
    current = [lines[0]]

    for i in range(1, len(lines)):
        prev = lines[i - 1]
        curr = lines[i]

        gap = curr["y"] - (prev["y"] + prev["h"])

        if gap > base_gap * gap_factor:
            groups.append(current)
            current = [curr]
        else:
            current.append(curr)

    groups.append(current)
    return groups


def _select_last_relevant_group(groups):
    return groups[-1] if groups else []


# =========================
# DETECTAR RESPOSTA
# =========================
def detect_marked_answer(question_img, option_lines, q_index=0, debug=True):
    debug_path = _ensure_debug_dir(q_index)

    results = []
    debug_img = question_img.copy()

    for idx, line in enumerate(option_lines):
        x, y, w, h = line["x"], line["y"], line["w"], line["h"]

        # 🔥 EXPANSÃO PARA ESQUERDA (CRÍTICO)
        expand_left = 0
        x_start = max(0, x - expand_left)

        roi = question_img[y:y+h, x_start:x+w]

        if roi.size == 0:
            continue

        roi_h, roi_w = roi.shape[:2]

        # 🔥 região da bolinha
        bubble = roi[
            int(roi_h * 0.2):int(roi_h * 0.8),
            int(roi_w * 0.08):int(roi_w * 0.22)
        ]

        if bubble.size == 0:
            continue

        gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        density = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])

        print(f"[Q{q_index}] opção {idx}: {density:.3f}")

        results.append({
            "index": idx,
            "density": density,
            "box": line
        })

        if debug:
            cv2.imwrite(f"{debug_path}/bubble_{idx}.png", bubble)
            cv2.imwrite(f"{debug_path}/bubble_thresh_{idx}.png", thresh)

            # desenha área analisada
            debug_roi = roi.copy()
            cv2.rectangle(
                debug_roi,
                (int(roi_w * 0.02), int(roi_h * 0.2)),
                (int(roi_w * 0.25), int(roi_h * 0.8)),
                (0, 0, 255),
                2
            )
            cv2.imwrite(f"{debug_path}/roi_{idx}.png", debug_roi)

    if not results:
        return None

    results = sorted(results, key=lambda r: r["density"], reverse=True)
    best = results[0]

    if best["density"] < 0.10:
        return None

    marked = [r for r in results if r["density"] > 0.18]
    if len(marked) > 1:
        return "MULTIPLE"

    letters = ["A", "B", "C", "D"]

    print(f"[Q{q_index}] resposta: {letters[best['index']]}")

    if debug:
        bx = best["box"]
        cv2.rectangle(
            debug_img,
            (bx["x"], bx["y"]),
            (bx["x"] + bx["w"], bx["y"] + bx["h"]),
            (0, 255, 0),
            3
        )
        cv2.imwrite(f"{debug_path}/answer.png", debug_img)

    return letters[best["index"]]


# =========================
# PIPELINE
# =========================
def detect_option_lines(question_img, q_index=0, debug=True) -> List[Dict]:
    debug_path = _ensure_debug_dir(q_index)

    if debug:
        cv2.imwrite(f"{debug_path}/0_original.png", question_img)

    candidates = _extract_line_candidates(
        question_img,
        debug_path=debug_path if debug else None
    )

    groups = _group_by_vertical_spacing(candidates)
    option_lines = _select_last_relevant_group(groups)

    if debug:
        debug_img = question_img.copy()

        for item in candidates:
            x, y, w, h = item["x"], item["y"], item["w"], item["h"]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 1)

        for item in option_lines:
            x, y, w, h = item["x"], item["y"], item["w"], item["h"]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite(f"{debug_path}/5_lines_filtered.png", debug_img)

    print(f"[Q{q_index}] grupos: {len(groups)}")
    print(f"[Q{q_index}] linhas finais: {len(option_lines)}")

    answer = detect_marked_answer(
        question_img,
        option_lines,
        q_index=q_index,
        debug=debug
    )

    return {
        "lines": option_lines,
        "answer": answer
    }
    