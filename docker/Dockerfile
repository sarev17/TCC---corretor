FROM python:3.11-slim

# =========================
# 1. Variáveis de ambiente
# =========================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# =========================
# 2. Dependências do sistema
# =========================
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# =========================
# 3. Diretório da aplicação
# =========================
WORKDIR /app

# =========================
# 4. Instalar dependências Python
# =========================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# =========================
# 5. Copiar aplicação
# =========================
COPY . .

# =========================
# 6. Expor porta
# =========================
EXPOSE 8000

# =========================
# 7. Subir aplicação
# =========================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]