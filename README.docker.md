# Docker Kurulumu Rehberi

## 📋 Gereksinimler

- **Docker Desktop** >= 20.10 (Windows/Mac) veya Docker Engine (Linux)
- **NVIDIA Docker Runtime** (GPU desteği için)
  - Windows: NVIDIA CUDA >= 11.8
  - Linux: `nvidia-docker2` paketi kurulu
- **Model dosyaları** mevcut:
  - `models/model.onnx` ✅
  - `models/latest.pt` (isteğe bağlı)

## 🚀 Hızlı Başlangıç

### 1. **GPU Desteğini Kontrol Et** (Linux/WSL2 için)

```bash
# Docker'ın GPU'yu görebilip göremediğini kontrol et
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Windows'ta Docker Desktop → Settings → Resources → GPUs açık mı diye kontrol et.

### 2. **Uygulamayı Başlat**

```bash
# Her iki servis'i başlat (backend + frontend)
docker-compose up --build

# Arka planda çalıştırmak için:
docker-compose up -d --build
```

### 3. **Servisler Erişilebilir mi?**

- **Backend API**: http://localhost:8000
  - Sağlık kontrolü: http://localhost:8000/health
- **Frontend Web**: http://localhost:5173
  - API baseUrl otomatik olarak `http://backend:8000` (container içinden)

---

## 📂 Yapı Açıklaması

```
├── api/docker/Dockerfile          # Backend (FastAPI + GPU)
├── frontend/Dockerfile            # Frontend (React/Vite)
├── docker-compose.yml             # Orkestrasyonu
├── .dockerignore                  # Exclude gereksiz dosyalar
├── models/
│   ├── model.onnx               # ✅ ONNX model
│   └── latest.pt                # (isteğe bağlı)
└── logs/                         # Çalışma zamanında oluşturulur
    └── processed_videos/
```

---

## 🔧 Komutlar

### Build ve Başlat
```bash
# Build + Start
docker-compose up --build

# Sadece rebuild
docker-compose build

# Detached mode (arka planda)
docker-compose up -d --build
```

### Stop/Restart
```bash
# Durdur
docker-compose down

# Restart
docker-compose restart

# Logs görüntüle
docker-compose logs -f backend    # Backend logs
docker-compose logs -f frontend   # Frontend logs
```

### Tekil Service

```bash
# Sadece backend'i başlat
docker-compose up backend

# Sadece frontend'i başlat  
docker-compose up frontend
```

---

## 🐛 Sorun Giderme

### **GPU görülmüyor**
```bash
# Check docker GPU availability
docker run --rm --gpus all ubuntu nvidia-smi

# docker-compose.yml'de GPU enabled mi?
# deploy.resources.reservations.devices.driver: nvidia
```

### **Model dosyası bulunamıyor**
```
Error: FileNotFoundError: /app/models/model.onnx
```
**Çözüm**: `models/` klasörü container'da mounted. Dosyaların host'ta olduğunu kontrol et:
```bash
ls -la models/
# model.onnx olmalı
```

### **Frontend API'ye bağlanamıyor**
```
Error: Failed to fetch from http://localhost:8000
```
**Çözüm**: Frontend container'dan `http://backend:8000` kullanıyor (DNS). Backend healthy mi?
```bash
docker-compose ps
docker-compose logs backend
```

### **Port çakışması**
```
Error: Port 8000 is already in use
```
**Çözüm**: docker-compose.yml'de port değiştir:
```yaml
ports:
  - "8001:8000"  # 8001'e yönlendir
```

---

## 📊 Performance & GPU

### Backend GPU Metrikleri
```bash
# Container içinde GPU kullanımını kontrol et
docker exec -it edge-ai-backend nvidia-smi

# Veya API'den
curl http://localhost:8000/metrics
```

### Memory Limitleri Ayarla
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

---

## 🔐 Production Ayarları

### Environment Variables (.env)
```bash
# .env dosyası oluştur
CUDA_VISIBLE_DEVICES=0
MODEL_BACKEND=onnx
VITE_API_URL=https://api.your-domain.com
PYTHONUNBUFFERED=1
```

### Reverse Proxy (Nginx)
```nginx
upstream backend {
    server backend:8000;
}
upstream frontend {
    server frontend:5173;
}

server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://backend/;
    }

    location / {
        proxy_pass http://frontend/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 💡 İpuçları

1. **Development Mode**:
   ```bash
   # Frontend'i hot-reload ile çalıştır
   # frontend/Dockerfile CMD'sini uncomment et: npm run dev
   docker-compose up frontend
   ```

2. **Logs Takibi**:
   ```bash
   docker-compose logs -f --tail=50
   ```

3. **Clean Up**:
   ```bash
   # Kullanılmayan images/containers temizle
   docker system prune -a
   ```

4. **Network Debugging**:
   ```bash
   # Aralarında bağlantı test et
   docker-compose exec frontend ping backend
   ```

---

## ✅ Kontrol Listesi

- [ ] Docker/Docker Compose kurulu
- [ ] NVIDIA Docker Runtime kurulu (GPU için)
- [ ] `models/model.onnx` mevcut
- [ ] Port 8000 ve 5173 boş
- [ ] `.dockerignore` dosyası var
- [ ] `docker-compose up --build` başarılı
- [ ] http://localhost:8000/health → `{"status":"ok"}`
- [ ] http://localhost:5173 erişilebilir
- [ ] GPU kullanılıyor (nvidia-smi)
