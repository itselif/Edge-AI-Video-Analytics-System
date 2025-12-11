# ============================================================================
# NVIDIA TensorRT Runtime base image (CUDA + cuDNN + TRT)
# For GPU-accelerated object detection with ONNX inference
# ============================================================================

FROM nvcr.io/nvidia/tensorrt:23.05-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy entire project (excluding files in .dockerignore)
# Context is project root, so everything copies correctly
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pillow \
    numpy \
    pydantic \
    opencv-python \
    onnxruntime-gpu \
    pynvml \
    python-dotenv

# Create directories for models and logs if they don't exist
RUN mkdir -p /app/models /app/logs /app/logs/processed_videos

# Expose API port
EXPOSE 8000

# Environment variables for GPU acceleration
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV MODEL_BACKEND="onnx"
ENV ONNX_PATH="/app/models/model.onnx"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Launch FastAPI backend
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
    