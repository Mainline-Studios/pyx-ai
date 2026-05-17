FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
RUN python -c 'from TTS.api import TTS; TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False); print("Tacotron model cache primed.")'

COPY . /app

CMD ["sh", "-lc", "gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --worker-class gthread --workers 1 --threads 16 --timeout 300 --graceful-timeout 60 --keep-alive 75"]
