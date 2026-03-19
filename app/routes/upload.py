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
        # 1. Carregar imagem
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

        pil_img = images[0].convert("RGB")
        img_np = np.array(pil_img)

        # =========================
        # 2. Segmentar questões
        # =========================
        question_regions = segment_questions(pil_img)

        all_boxes = []

        # =========================
        # 3. Boxes das questões (VERDE)
        # =========================
        for r in question_regions:
            all_boxes.append({
                **r,
                "color": (0, 255, 0)
            })

        # =========================
        # 4. Detectar ITENS (AZUL)
        # =========================
        for q_index, r in enumerate(question_regions):
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]

            if y + h > img_np.shape[0] or x + w > img_np.shape[1]:
                continue

            question_img = img_np[y:y+h, x:x+w]

            # 🔥 NOVO: detecta linhas + salva debug
            option_lines = detect_option_lines(
                question_img,
                q_index=q_index,
                debug=True
            )

            # 🔥 desenhar cada alternativa
            for line in option_lines:
                all_boxes.append({
                    "x": x + line["x"],
                    "y": y + line["y"],
                    "w": line["w"],
                    "h": line["h"],
                    "color": (255, 0, 0)  # azul
                })

        # =========================
        # 5. Render final
        # =========================
        debug_img = draw_boxes(pil_img, all_boxes)

        # =========================
        # 6. Retorno
        # =========================
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