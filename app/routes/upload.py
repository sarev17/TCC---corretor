from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

from app.services.draw_service import draw_boxes
from app.services.pdf_service import pdf_to_images
from app.services.image_service import detect_regions

from PIL import Image
import cv2
import io

router = APIRouter(prefix="/upload")  # 👈 ISSO FALTAVA

@router.post("/preview")
async def preview_exam(file: UploadFile = File(...)):

    contents = await file.read()
    filename = file.filename.lower()

    images = []

    if filename.endswith(".pdf"):
        with open("temp.pdf", "wb") as f:
            f.write(contents)
        images = pdf_to_images("temp.pdf")

    else:
        image = Image.open(io.BytesIO(contents))
        images = [image]

    # pegar só a primeira página por enquanto
    img = images[0]

    boxes = detect_regions(img)

    img_with_boxes = draw_boxes(img, boxes)

    # converter para jpg
    _, buffer = cv2.imencode(".jpg", img_with_boxes)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )