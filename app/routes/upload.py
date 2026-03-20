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
        # 1. Carregar imagens
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

        all_pages = []

        # =========================
        # 🔥 PROCESSAR TODAS AS PÁGINAS
        # =========================
        for page_index, pil_img in enumerate(images):

            print("\n========================")
            print(f"[PÁGINA] {page_index}")
            print("========================")

            pil_img = pil_img.convert("RGB")
            img_np = np.array(pil_img)

            question_regions = segment_questions(pil_img)

            all_boxes = []
            answers = []

            # caixas das questões
            for r in question_regions:
                all_boxes.append({
                    **r,
                    "color": (0, 255, 0)
                })

            # =========================
            # PROCESSAR QUESTÕES
            # =========================
            for q_index, r in enumerate(question_regions):

                x, y, w, h = r["x"], r["y"], r["w"], r["h"]

                if y + h > img_np.shape[0] or x + w > img_np.shape[1]:
                    continue

                question_img = img_np[y:y+h, x:x+w]

                result = detect_option_lines(
                    question_img,
                    q_index=q_index,
                    debug=True
                )

                option_lines = result["lines"]
                answer = result["answer"]

                answers.append({
                    "question": q_index,
                    "answer": answer
                })

                print(f"[RESULTADO] Q{q_index}: {answer}")

                # desenhar alternativas
                for line in option_lines:
                    all_boxes.append({
                        "x": x + line["x"],
                        "y": y + line["y"],
                        "w": line["w"],
                        "h": line["h"],
                        "color": (255, 0, 0)
                    })

                # desenhar resposta
                if answer and answer != "MULTIPLE":
                    cv2.putText(
                        img_np,
                        f"{answer}",
                        (x + w - 50, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )

                elif answer == "MULTIPLE":
                    cv2.putText(
                        img_np,
                        "M",
                        (x + w - 50, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )

            debug_img = draw_boxes(Image.fromarray(img_np), all_boxes)

            all_pages.append(debug_img)

        # =========================
        # 🔥 JUNTAR TODAS AS PÁGINAS
        # =========================
        final_image = np.vstack(all_pages)

        _, buffer = cv2.imencode(".jpg", final_image)

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