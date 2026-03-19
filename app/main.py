from fastapi import FastAPI
from app.routes.upload import router as upload_router

app = FastAPI(title="Exam Corrector API")

app.include_router(upload_router)