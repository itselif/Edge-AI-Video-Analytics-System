# Backend Docker Setup

## 📋 Gereksinimler

- Docker Desktop >= 20.10
- NVIDIA Docker Runtime (GPU desteği için)
- `models/model.onnx` dosyası mevcut

## 🐳 Build & Run

### Tek başına (test için):
```bash
cd api/docker
docker build -t edge-ai-backend .
docker run --gpus all -p 8000:8000 edge-ai-backend
```

### Docker Compose ile (önerilen):
```bash
cd ../..  # Project root
docker-compose up backend
```

## 🔍 Kontrol

```bash
# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics

# Debug jobs
curl http://localhost:8000/debug/jobs
```

## 📝 Environment Variables

```
PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0
MODEL_BACKEND=onnx
ONNX_PATH=/app/models/model.onnx
```

## 📂 Volume Mounts

- `/app/models` → Host `./models`
- `/app/logs` → Host `./logs`

## 🚀 GPU Support

GPU'yu etkinleştirmek için:
```bash
docker run --gpus all ...
```

Dockerfile'da GPU otomatik olarak desteklenmektedir.
