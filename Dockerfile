FROM python:3.11-slim

# Install system dependencies required by dlib, face_recognition, and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Railway dynamically assigns PORT - do NOT hardcode or EXPOSE a specific port
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
