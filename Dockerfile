FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=5000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    fontconfig \
    fonts-dejavu-core \
    ghostscript \
    libcairo2 \
    libcairo2-dev \
    libglib2.0-0 \
    libgl1 \
    libice6 \
    libpango1.0-0 \
    libpango1.0-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    lmodern \
    pkg-config \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-science \
    texlive-xetex \
    dvisvgm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY PageIndex/requirements.txt ./PageIndex/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt -r PageIndex/requirements.txt pymongo

COPY backend/ ./backend/
COPY PageIndex/ ./PageIndex/
COPY data/models/piper/ ./backend/data/models/piper/

WORKDIR /app/backend

EXPOSE 5000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]
