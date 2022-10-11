# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501 8000

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "stable_diffusion_app.py", "--server.port=8501", "--server.address=0.0.0.0"]