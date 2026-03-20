from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

from app.services.segmentation_service import segment_questions
from app.services.draw_service import draw_boxes
from app.services.answer_visual_service import detect_option_lines
from app.services.pdf_service import pdf_to_images

from PIL import Image
import cv2
import numpy as np
import io
import os

router = APIRouter(prefix="/upload")


@router.post("/preview")
async def preview_exam(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        filename = file.filename.lower()

        # =========================
        # CARREGAR IMAGENS
        # =========================
        if filename.endswith(".pdf"):
            temp_path = "temp.pdf"

            with open(temp_path, "wb") as f:
                f.write(contents)

            images = pdf_to_images(temp_path)

            if os.path.exists(temp_path):
                os.remove(temp_path)

        else:
            image = Image.open(io.BytesIO(contents))
            images = [image]

        # =========================
        # 🔥 UNIFICAR TODAS AS PÁGINAS
        # =========================
        stacked_pages = []

        max_width = 0

        for pil_img in images:
            pil_img = pil_img.convert("RGB")
            img_np = np.array(pil_img)

            cropped = crop_vertical_whitespace(img_np)

            stacked_pages.append(cropped)
            
            if img_np.shape[1] > max_width:
                max_width = img_np.shape[1]

        # 🔥 normalizar largura
        normalized_pages = []
        for img in stacked_pages:
            if img.shape[1] != max_width:
                img = cv2.resize(img, (max_width, img.shape[0]))
            normalized_pages.append(img)

        # 🔥 junta tudo
        full_image = np.vstack(normalized_pages)

        print("\n========================")
        print("[PROCESSANDO IMAGEM ÚNICA]")
        print("========================")

        # =========================
        # SEGMENTAÇÃO GLOBAL
        # =========================
        question_regions = segment_questions(Image.fromarray(full_image))

        print(f"[TOTAL QUESTÕES]: {len(question_regions)}")

        all_boxes = []

        # =========================
        # PROCESSAMENTO GLOBAL
        # =========================
        for q_index, r in enumerate(question_regions):

            x, y, w, h = r["x"], r["y"], r["w"], r["h"]

            if y + h > full_image.shape[0] or x + w > full_image.shape[1]:
                continue

            # =========================
            # DEBUG: NUMERO DA QUESTÃO
            # =========================
            cv2.putText(
                full_image,
                f"Q{q_index}",
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2
            )

            question_img = full_image[y:y+h, x:x+w]

            result = detect_option_lines(
                question_img,
                q_index=q_index,
                debug=True
            )

            option_lines = result["lines"]
            answer = result["answer"]

            print(f"[RESULTADO] Q{q_index}: {answer}")

            # =========================
            # DESENHAR REGIÃO DA QUESTÃO
            # =========================
            all_boxes.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "color": (0, 255, 0)
            })

            # =========================
            # DESENHAR ALTERNATIVAS
            # =========================
            for line in option_lines:
                all_boxes.append({
                    "x": x + line["x"],
                    "y": y + line["y"],
                    "w": line["w"],
                    "h": line["h"],
                    "color": (255, 0, 0)
                })

            # =========================
            # DESENHAR RESPOSTA
            # =========================
            if answer and answer != "MULTIPLE":
                cv2.putText(
                    full_image,
                    f"{answer}",
                    (x + w - 50, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

            elif answer == "MULTIPLE":
                cv2.putText(
                    full_image,
                    "M",
                    (x + w - 50, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

        # =========================
        # OUTPUT FINAL
        # =========================
        debug_img = draw_boxes(Image.fromarray(full_image), all_boxes)

        _, buffer = cv2.imencode(".jpg", debug_img)

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "error": str(e),
            "type": str(type(e))
        }


def crop_vertical_whitespace(img, threshold=250, padding=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    row_sum = np.sum(thresh, axis=1)

    non_empty_rows = np.where(row_sum > 0)[0]

    if len(non_empty_rows) == 0:
        return img

    top = max(0, non_empty_rows[0] - padding)
    bottom = min(img.shape[0], non_empty_rows[-1] + padding)

    return img[top:bottom]