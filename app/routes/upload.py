from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

from app.services.segmentation_service import segment_questions
from app.services.draw_service import draw_boxes
from app.services.pdf_service import pdf_to_images  # 👈 IMPORTANTE

from PIL import Image
import cv2
import io

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
            with open("temp.pdf", "wb") as f:
                f.write(contents)

            images = pdf_to_images("temp.pdf")  # 👈 DESCOMENTA
        else:
            image = Image.open(io.BytesIO(contents))
            images = [image]

        pil_img = images[0]

        # =========================
        # 2. Segmentar questões
        # =========================
        question_regions = segment_questions(pil_img)

        # =========================
        # 3. Debug
        # =========================
        debug_img = draw_boxes(pil_img, question_regions, (0, 255, 0))

        # =========================
        # 4. Converter saída
        # =========================
        _, buffer = cv2.imencode(".jpg", debug_img)

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {"error": str(e)}