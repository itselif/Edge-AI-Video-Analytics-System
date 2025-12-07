# ---------------------------------------------------------
# NVIDIA TensorRT Runtime base image (CUDA + cuDNN + TRT)
# ---------------------------------------------------------
    FROM nvcr.io/nvidia/tensorrt:23.05-py3

    # Set working directory
    WORKDIR /app
    
    # Copy project
    COPY .. /app/
    
    # Install Python dependencies
    RUN pip install --upgrade pip && \
        pip install fastapi uvicorn[standard] python-multipart pillow numpy onnxruntime-gpu pynvml
    
    # Expose API port
    EXPOSE 8000
    
    # Environment variables
    ENV MODEL_PATH="/app/models/model_fp16.engine"
    ENV ONNX_PATH="/app/models/model.onnx"
    ENV CUDA_DEVICE=0
    
    # Launch API
    CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
    