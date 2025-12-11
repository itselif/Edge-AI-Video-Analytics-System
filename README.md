# Edge AI Video Analytics System: Multi-Backend Object Detection & Tracking

This project is a complete, end-to-end Edge AI Video Analytics Solution designed for real-time object detection and tracking on resource-constrained devices. It features a flexible **Multi-Backend Inference Engine** that automatically selects the best available execution provider—ranging from pure CPU execution to high-performance GPU acceleration using **NVIDIA TensorRT**.

## Dataset & Model Information

The model targets traffic and pedestrian analysis scenarios using a specific subset of the **COCO (Common Objects in Context)** dataset.

**Model Architecture:** `YOLO11n` (Nano)
**Selected Classes:**
The system is trained to detect the following 5 classes:
1.  `person`
2.  `bicycle`
3.  `car`
4.  `motorcycle`
5.  `bus`

**Data Augmentation:**
To ensure robustness against varying lighting and environmental conditions, strong augmentations were applied during training using **Albumentations** (Mosaic, MixUp, Color Jitter, Motion Blur).

---

## Tech Stack

The system is designed for maximum compatibility and performance.

| Category | Technologies |
| :--- | :--- |
| **Inference Backends** | **ONNX Runtime (CPU & CUDA)**, **NVIDIA TensorRT (FP16/INT8)**, PyTorch |
| **AI Architecture** | YOLO11, ByteTrack/DeepSORT (Hybrid Tracking) |
| **Backend API** | **FastAPI**, Uvicorn, WebSockets (Real-time streaming) |
| **Frontend UI** | **React (v18)**, TypeScript, Vite, Tailwind CSS, Recharts |
| **Deployment** | Docker, NVIDIA Container Toolkit (Optional for GPU) |

---

## System Architecture & Workflow

The solution is divided into four main stages, combining high-performance optimization with architectural flexibility:

### 1. Model Training
The YOLO11n model was trained on the custom COCO subset. Techniques such as **EMA (Exponential Moving Average)** and **AMP (Automatic Mixed Precision)** were utilized to stabilize training and improve convergence.

### 2. Optimization Pipeline (Edge Performance & Compatibility)
To ensure the model runs efficiently on *any* hardware while unlocking maximum speed on NVIDIA GPUs, a two-stage pipeline is used:
1.  **PyTorch to ONNX (Universal Access):** The trained `.pt` model is exported to ONNX format. This enables the system to run on standard CPUs and CUDA environments without requiring specialized drivers.
2.  **TensorRT Conversion (The Edge Factor):** For NVIDIA deployment, the ONNX model is compiled into a highly optimized **TensorRT Engine (.engine)**.
3.  **Quantization:** FP16 (Half-Precision) and INT8 quantization are applied to significantly reduce memory footprint and maximize inference speed (FPS) with minimal accuracy loss.

### 3. Adaptive Hybrid Inference Engine
The system employs a smart `Detector` class that automatically selects the best execution provider (TensorRT > CUDA > CPU). It utilizes a **Detector + Tracker** fusion strategy:
* **Detection:** Runs periodically (every N frames) to identify objects using the selected backend.
* **Tracking:** A lightweight tracker follows objects between detection frames to maintain real-time fluidity (especially crucial for CPU execution).
* **Drift Correction:** An IoU-based logic constantly monitors the tracker; if the tracker drifts, the detector re-initializes the object.

### 4. Real-Time Visualization
* The Backend processes the video feed and streams processed frames and metadata via **WebSockets**.
* The Frontend (React) displays the live video, bounding boxes, and real-time performance metrics (FPS, Latency, GPU Usage).

---

## Performance Objectives

By leveraging TensorRT optimization alongside a multi-backend architecture, this system achieves:

* **Low Latency:** Minimized pre/post-processing time using optimized engines.
* **High Throughput:** Batch processing support for handling multiple streams on GPUs.
* **Efficiency:** Reduced GPU memory consumption via INT8/FP16 precision.
* **Adaptability:** Seamless fallback to ONNX Runtime (CPU) ensures the system remains functional even without dedicated hardware.
## Installation & Usage

You can run the system on any machine with Docker installed.

### Prerequisites
* **Docker & Docker Compose**
* *(Optional)* NVIDIA GPU Driver & Container Toolkit (Required only for TensorRT/CUDA modes)

### Quick Start

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/itselif/Edge-AI-Video-Analytics-System.git]
    cd edge-ai-video-analytics
    ```

2.  **Run with Docker**
    The system will automatically detect if a GPU is available.
    ```bash
    docker-compose up --build
    ```
    *If running on CPU-only, ensure the Dockerfile is set to use the standard runtime.*

3.  **Access the Dashboard**
    Open your browser and navigate to `http://localhost:5173` to view the live detection stream and switch between inference modes.

---

## License

This project is open-source and available under the **MIT License**.
