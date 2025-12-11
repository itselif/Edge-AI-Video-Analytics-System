from __future__ import annotations

import io
import time
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
import threading
import mimetypes

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import subprocess
import cv2

from inference.detector import Detector, Detection
from monitoring.logger import LatencyMeter, JsonLogger, get_gpu_stats
from api.schemas import (
    HealthResponse,
    DetectResponse,
    BBox,
    MetricsResponse,
)

# 5-class COCO dataset
COCO_5CLS = ["person", "bicycle", "car", "motorcycle", "bus"]

# Project directories
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = LOG_DIR / "processed_videos"
PROCESSED_DIR.mkdir(exist_ok=True)
DEFAULT_BACKEND = "onnx"
DEFAULT_MODEL_ONNX = MODELS_DIR / "model.onnx"

# FastAPI application
app = FastAPI(title="Edge AI Video Analytics API")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Static file service: processed_videos folder
app.mount("/processed_videos", StaticFiles(directory=PROCESSED_DIR), name="processed_videos")

# Monitoring and logging
METRICS = LatencyMeter(window_size=100)
EVENT_LOGGER = JsonLogger(str(LOG_DIR / "api_events.jsonl"))

# Detector object
detector: Optional[Detector] = None

# Simple job store for async video processing
JOBS: dict = {}
JOBS_LOCK = threading.Lock()


@app.on_event("startup")
def startup_event() -> None:
    """Initialize model on service startup."""
    global detector

    detector = Detector(
        backend=DEFAULT_BACKEND,
        model_path=DEFAULT_MODEL_ONNX,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        device="cuda",
        class_names=COCO_5CLS,
    )

    print(f"[API] Loaded detector backend={DEFAULT_BACKEND}, model={DEFAULT_MODEL_ONNX}")
    print(f"[API] Processed videos directory: {PROCESSED_DIR}")


@app.on_event("shutdown")
def shutdown_event() -> None:
    """Cleanup on shutdown."""
    if EVENT_LOGGER:
        EVENT_LOGGER.close()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Basic health-check endpoint."""
    return HealthResponse(
        status="ok",
        backend=DEFAULT_BACKEND,
        model_path=str(DEFAULT_MODEL_ONNX),
        detail="Service healthy",
    )


def _detections_to_bboxes(det_list: List[Detection]) -> List[BBox]:
    """Convert internal Detection objects to API BBox schema."""
    boxes: List[BBox] = []

    for d in det_list:
        cls_id = int(d.cls)
        label = None

        if detector and detector.class_names:
            if 0 <= cls_id < len(detector.class_names):
                label = detector.class_names[cls_id]

        boxes.append(
            BBox(
                x1=float(d.x1),
                y1=float(d.y1),
                x2=float(d.x2),
                y2=float(d.y2),
                score=float(d.score),
                cls_id=cls_id,
                label=label,
            )
        )

    return boxes


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    """Run object detection on a single image."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector is not initialized")

    raw_bytes = await file.read()
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    frame = np.array(img)[:, :, ::-1]  # RGB → BGR

    t0 = time.time()
    det_list: List[Detection] = detector(frame)
    latency_ms = (time.time() - t0) * 1000.0

    METRICS.record_latency(latency_ms)

    bboxes = _detections_to_bboxes(det_list)

    EVENT_LOGGER.log(
        "detect",
        {
            "latency_ms": float(latency_ms),
            "num_detections": len(bboxes),
        },
    )

    return DetectResponse(
        backend=DEFAULT_BACKEND,
        model_path=str(DEFAULT_MODEL_ONNX),
        inference_time_ms=float(latency_ms),
        num_detections=len(bboxes),
        detections=bboxes,
    )


@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    """
    Fast video inference pipeline using pure OpenCV:
    - Reads video frames directly
    - Runs GPU-accelerated detection
    - Draws bounding boxes on each frame
    - Re-encodes output using vp80 (WebM) for browser support
    """

    if detector is None:
        raise HTTPException(status_code=503, detail="Detector is not initialized")

    suffix = Path(file.filename).suffix or ".mp4"

    # UPDATE 1: Change output file extension to .webm
    tmp_input = Path(tempfile.gettempdir()) / f"in_{uuid.uuid4().hex}{suffix}"
    tmp_output = PROCESSED_DIR / f"out_{uuid.uuid4().hex}.webm"

    try:
        # Save uploaded file
        with tmp_input.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        cap = cv2.VideoCapture(str(tmp_input))
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Failed to open input video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # UPDATE 2: Change codec to WebM (VP8)
        # This format works smoothly on Chrome/Firefox/Edge
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        writer = cv2.VideoWriter(str(tmp_output), fourcc, fps, (width, height))

        if not writer.isOpened():
            raise HTTPException(status_code=500, detail="Failed to create output video")

        print(f"[API] Processing video: {tmp_input}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector(frame)

            # Draw detection boxes
            for d in detections:
                cv2.rectangle(
                    frame,
                    (int(d.x1), int(d.y1)),
                    (int(d.x2), int(d.y2)),
                    (0, 255, 0),
                    2,
                )

                if detector.class_names and 0 <= int(d.cls) < len(detector.class_names):
                    label = detector.class_names[int(d.cls)]
                    cv2.putText(
                        frame,
                        f"{label} {d.score:.2f}",
                        (int(d.x1), int(d.y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            writer.write(frame)

        cap.release()
        writer.release()

        if not tmp_output.exists():
            raise HTTPException(status_code=500, detail="Output video not generated")

        # UPDATE 3: Change response type to video/webm
        return FileResponse(
            path=str(tmp_output),
            media_type="video/webm",
            filename="processed.webm",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    except Exception as e:
        # Log error and inform user in case of failure
        print(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if tmp_input.exists():
                tmp_input.unlink()
        except:
            pass

def _process_video_job(job_id: str, input_path: str, output_path: str) -> None:
    """Background worker: process video and update JOBS progress."""
    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["start_time"] = time.time()

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open input video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Store initial fps so frontend can read it during processing
        with JOBS_LOCK:
            JOBS[job_id]["fps"] = fps


        # CRITICAL FIX: Use WebM format (vp80 codec)
        # Replace .mp4 with .webm extension
        webm_output = output_path.replace('.mp4', '.webm')

        # Use VP8 codec for WebM
        fourcc = cv2.VideoWriter_fourcc(*'vp80')

        # Alternative: VP9 codec (better compression)
        # fourcc = cv2.VideoWriter_fourcc(*'vp90')

        
        writer = cv2.VideoWriter(str(webm_output), fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"[API] Warning: 'vp80' codec failed, trying 'mp4v' as fallback")
            # Fallback: eski mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            webm_output = output_path 
            writer = cv2.VideoWriter(str(webm_output), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Failed to create output video with any codec")

        processed = 0
        
        # Define different colors for each class
        class_colors = {
            0: (0, 0, 255),      # person - red
            1: (255, 0, 0),      # bicycle - blue
            2: (0, 255, 0),      # car - green
            3: (255, 255, 0),    # motorcycle - cyan
            4: (255, 0, 255),    # bus - magenta
        }
        
        default_color = (0, 255, 0)

        print(f"[API] Starting video processing: {total_frames} frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector(frame)

            # Update detection count in job metadata so UI can show live numbers
            try:
                with JOBS_LOCK:
                    JOBS[job_id]["detection_count"] = len(detections)
            except Exception:
                pass

            for d in detections:
                cls_id = int(d.cls)
                color = class_colors.get(cls_id, default_color)
                
                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (int(d.x1), int(d.y1)),
                    (int(d.x2), int(d.y2)),
                    color,
                    2,
                )

                # Draw label
                if detector.class_names and 0 <= cls_id < len(detector.class_names):
                    label = detector.class_names[cls_id]
                    score_text = f"{label} {d.score:.2f}"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    (text_width, text_height), baseline = cv2.getTextSize(
                        score_text, font, font_scale, thickness
                    )
                    
                    # Draw background
                    cv2.rectangle(
                        frame,
                        (int(d.x1), int(d.y1) - text_height - 10),
                        (int(d.x1) + text_width, int(d.y1)),
                        color,
                        -1,
                    )
                    
                    # Draw text
                    cv2.putText(
                        frame,
                        score_text,
                        (int(d.x1), int(d.y1) - 5),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                    )

            writer.write(frame)
            processed += 1
            
            # Update progress every 10 frames or every ~5%
            if processed % 10 == 0 or processed == total_frames:
                if total_frames > 0:
                    progress_percent = (processed / total_frames) * 100
                    with JOBS_LOCK:
                        JOBS[job_id]["progress"] = progress_percent
                        

                    if processed > 0:
                        elapsed = time.time() - JOBS[job_id]["start_time"]
                        if processed > 10: 
                            remaining = max(0.0, (total_frames - processed) * (elapsed / processed))
                            with JOBS_LOCK:
                                JOBS[job_id]["eta_seconds"] = remaining
                    
                    # Debug log
                    if processed % 50 == 0:
                        print(f"[API] Processed {processed}/{total_frames} frames ({progress_percent:.1f}%)")

        cap.release()
        writer.release()
        
        print(f"[API] Video writing completed, releasing writer...")
        output_file = Path(webm_output)
        if not output_file.exists():
            raise RuntimeError("Output video file was not created")
            
        file_size = output_file.stat().st_size
        if file_size == 0:
            raise RuntimeError("Output video file is empty (0 bytes)")

        print(f"[API] Video processing completed: {job_id}")
        print(f"[API] Output file: {webm_output}")
        print(f"[API] Output file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

        # Video metadata check
        test_cap = cv2.VideoCapture(str(webm_output))
        if test_cap.isOpened():
            test_fps = test_cap.get(cv2.CAP_PROP_FPS)
            test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            test_cap.release()
            print(f"[API] Video metadata: {test_frames} frames @ {test_fps} FPS")
        else:
            print(f"[API] Warning: Cannot read output video metadata")

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["progress"] = 100.0
            JOBS[job_id]["end_time"] = time.time()
            JOBS[job_id]["output_path"] = str(webm_output)  # WebM yolunu kaydet
            JOBS[job_id]["file_size"] = file_size
            JOBS[job_id]["format"] = "webm" if 'webm' in webm_output else "mp4"
            JOBS[job_id]["fps"] = fps

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
        print(f"[API] Video processing failed for {job_id}: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # cleanup input file
        try:
            p = Path(input_path)
            if p.exists():
                p.unlink()
        except Exception as e:
            print(f"[API] Failed to cleanup input file: {str(e)}")
            
@app.post("/detect_video_async")
async def detect_video_async(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Accept video, start background processing and return a job id immediately."""

    if detector is None:
        raise HTTPException(status_code=503, detail="Detector is not initialized")

    suffix = Path(file.filename).suffix or ".mp4"

    tmp_input = Path(tempfile.gettempdir()) / f"in_{uuid.uuid4().hex}{suffix}"

    try:
        # Save uploaded file
        with tmp_input.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        job_id = uuid.uuid4().hex
        tmp_output = PROCESSED_DIR / f"{job_id}.mp4"

        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "queued",
                "progress": 0.0,
                "start_time": None,
                "end_time": None,
                "eta_seconds": None,
                "input_path": str(tmp_input),
                "output_path": str(tmp_output),
                "error": None,
                "file_size": 0,
            }

        # Schedule background work
        if background_tasks is None:
            background_tasks = BackgroundTasks()

        background_tasks.add_task(_process_video_job, job_id, str(tmp_input), str(tmp_output))

        return {
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/detect_video_status/{job_id}",
            "download_url": f"/processed/{job_id}",
            "message": "Video processing started in background"
        }

    except Exception as e:
        # cleanup input if saved
        try:
            if tmp_input.exists():
                tmp_input.unlink()
        except:
            pass

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detect_video_status/{job_id}")
def detect_video_status(job_id: str):
    """Return current job status and progress."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        response = {
            "job_id": job_id,
            "status": job.get("status"),
            "progress": job.get("progress", 0.0),
            "eta_seconds": job.get("eta_seconds"),
            "error": job.get("error"),
            "detection_count": job.get("detection_count", 0),
            "fps": job.get("fps", 30),
            "download_url": (f"/processed/{job_id}" if job.get("status") == "done" else None),
        }
        
        # Add output path for debugging
        if job.get("status") == "done":
            output_path = job.get("output_path")
            if output_path:
                response["output_path"] = output_path
                # Check if file exists and add size
                if Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size
                    response["file_size"] = file_size
                    response["file_size_mb"] = file_size / (1024 * 1024)
        
        return response


@app.get("/processed/{job_id}")
def get_processed_video(job_id: str, request: Request):
    """Serve the processed video for a finished job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.get("status") != "done":
            raise HTTPException(status_code=409, detail="Job not finished")

        out_path = job.get("output_path")

    if not out_path:
        raise HTTPException(status_code=404, detail="Output path not defined")

    file_path = Path(out_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    # Optionally, you can use StreamingResponse for range requests
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=file_path.name,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/download_video/{job_id}")
async def download_video_to_path(job_id: str):
    """Download processed video to a specific path."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.get("status") != "done":
            raise HTTPException(status_code=409, detail="Job not finished")

        out_path = job.get("output_path")

    if not out_path:
        raise HTTPException(status_code=404, detail="Output path not defined")

    file_path = Path(out_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    dest_dir = Path("/teamspace/studios/this_studio/Edge-AI-Video-Analytics-System")
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / f"processed_{job_id}.mp4"

    try:
        shutil.copy2(str(file_path), str(dest_path))
        return {
            "message": f"Video downloaded to {dest_path}",
            "source": str(file_path),
            "destination": str(dest_path),
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy file: {str(e)}")


@app.post("/regenerate_video/{job_id}")
async def regenerate_video(job_id: str, background_tasks: BackgroundTasks = None):
    """Manually regenerate video for a job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Reset job status
        job["status"] = "processing"
        job["progress"] = 0.0
        job["error"] = None

        input_path = job.get("input_path")
        output_path = job.get("output_path")

    if not input_path or not Path(input_path).exists():
        raise HTTPException(status_code=404, detail="Input file not found")

    if background_tasks is None:
        background_tasks = BackgroundTasks()

    background_tasks.add_task(_process_video_job, job_id, input_path, output_path)

    return {
        "message": f"Video regeneration started for job {job_id}",
        "status": "processing"
    }


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Return latency + GPU metrics."""
    stats = METRICS.get_stats()
    gpu = get_gpu_stats()

    return MetricsResponse(
        backend=DEFAULT_BACKEND,
        model_path=str(DEFAULT_MODEL_ONNX),
        avg_latency_ms=stats["avg_latency_ms"],
        moving_avg_latency_ms=stats["moving_avg_latency_ms"],
        p50_latency_ms=stats["p50_latency_ms"],
        p90_latency_ms=stats["p90_latency_ms"],
        p95_latency_ms=stats["p95_latency_ms"],
        fps=stats["fps"],
        total_requests=stats["total_requests"],
        gpu_name=gpu["gpu_name"],
        gpu_memory_used_mb=gpu["gpu_memory_used_mb"],
        gpu_memory_total_mb=gpu["gpu_memory_total_mb"],
        gpu_utilization=gpu["gpu_utilization"],
    )

@app.get("/debug/jobs")
def debug_jobs():
    """Debug endpoint to see all jobs."""
    with JOBS_LOCK:
        return {
            "total_jobs": len(JOBS),
            "jobs": {
                job_id: {
                    "status": job.get("status"),
                    "output_path": job.get("output_path"),
                    "output_exists": Path(job.get("output_path", "")).exists() if job.get("output_path") else False,
                    "file_size": job.get("file_size"),
                    "progress": job.get("progress"),
                    "error": job.get("error"),
                }
                for job_id, job in JOBS.items()
            }
        }


@app.get("/debug/files")
def debug_files():
    """List all processed video files."""
    files = []
    for f in PROCESSED_DIR.glob("*"):
        files.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "size_mb": f.stat().st_size / (1024 * 1024),
            "modified": f.stat().st_mtime
        })
    
    return {
        "directory": str(PROCESSED_DIR),
        "total_files": len(files),
        "files": files
    }


@app.get("/check_video/{job_id}")
async def check_video_file(job_id: str):
    """Check if video file exists and get details."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return {"error": "Job not found"}
        
        out_path = job.get("output_path")
    
    if not out_path:
        return {"error": "Output path not defined"}
    
    file_path = Path(out_path)
    
    if not file_path.exists():
        return {"error": f"File does not exist: {out_path}"}
    
    # Try to read video metadata with OpenCV
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return {"error": "OpenCV cannot open video file"}
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Convert fourcc code to string
        fourcc_str = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        file_size = file_path.stat().st_size
        
        return {
            "exists": True,
            "file_size": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "fourcc": fourcc_str,
            "fourcc_code": fourcc_code,
            "path": str(file_path),
            "supported_by_browser": fourcc_str in ["mp4v", "avc1", "h264", "H264"]
        }
    except Exception as e:
        return {"error": f"Failed to read video metadata: {str(e)}"}


@app.post("/convert_video/{job_id}")
async def convert_video_format(job_id: str):
    """Convert video to browser-compatible format."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.get("status") != "done":
            raise HTTPException(status_code=409, detail="Job not finished")
        
        out_path = job.get("output_path")
    
    if not out_path:
        raise HTTPException(status_code=404, detail="Output path not defined")
    
    file_path = Path(out_path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Create converted file path
    converted_path = PROCESSED_DIR / f"{job_id}_converted.mp4"
    
    try:
        # Open the original video
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open original video")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Try different codecs for browser compatibility
        codecs_to_try = [
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
            ('H264', cv2.VideoWriter_fourcc(*'H264')),  # Alternative H.264
        ]
        
        writer = None
        selected_codec = None
        
        for codec_name, fourcc in codecs_to_try:
            writer = cv2.VideoWriter(str(converted_path), fourcc, fps, (width, height))
            if writer.isOpened():
                selected_codec = codec_name
                print(f"[API] Using codec: {codec_name}")
                break
            else:
                writer = None
        
        if writer is None:
            cap.release()
            raise HTTPException(status_code=500, detail="Cannot create video writer with any codec")
        
        print(f"[API] Converting video to browser-compatible format: {converted_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        
        cap.release()
        writer.release()
        
        # Check if converted file exists and has size
        if not converted_path.exists():
            raise HTTPException(status_code=500, detail="Converted file not created")
        
        converted_size = converted_path.stat().st_size
        if converted_size == 0:
            raise HTTPException(status_code=500, detail="Converted file is empty")
        
        print(f"[API] Conversion successful: {converted_size} bytes")
        
        # Update job with converted path
        with JOBS_LOCK:
            job["converted_path"] = str(converted_path)
            job["converted_url"] = f"/converted/{job_id}"
        
        return {
            "message": "Video converted to browser-compatible format",
            "original_path": str(file_path),
            "converted_path": str(converted_path),
            "converted_url": f"/converted/{job_id}",
            "codec": selected_codec,
            "file_size_mb": converted_size / (1024 * 1024)
        }
        
    except Exception as e:
        # Cleanup if conversion failed
        if converted_path.exists():
            try:
                converted_path.unlink()
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Video conversion failed: {str(e)}")


@app.get("/converted/{job_id}")
def get_converted_video(job_id: str):
    """Serve the converted video."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        converted_path = job.get("converted_path")
    
    if not converted_path:
        raise HTTPException(status_code=404, detail="Converted video not available")
    
    file_path = Path(converted_path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Converted file not found")
    
    return _range_response_for_file(file_path, request)


@app.get("/stream_video/{job_id}")
def stream_video(job_id: str, request: Request):
    """Stream video with proper headers for browser playback."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.get("status") != "done":
            raise HTTPException(status_code=409, detail="Job not finished")

        out_path = job.get("output_path")

    if not out_path:
        raise HTTPException(status_code=404, detail="Output path not defined")

    file_path = Path(out_path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return _range_response_for_file(file_path, request)



@app.get("/test_video/{job_id}")
def test_video_playback(job_id: str):
    """Return HTML page to test video playback directly."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.get("status") != "done":
            raise HTTPException(status_code=409, detail="Job not finished")

        out_path = job.get("output_path")

    if not out_path:
        raise HTTPException(status_code=404, detail="Output path not defined")

    file_path = Path(out_path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    video_url = f"/processed/{job_id}"
    stream_url = f"/stream_video/{job_id}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Test - Job {job_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            .video-container {{ margin: 20px 0; }}
            video {{ max-width: 100%; border: 1px solid #ccc; }}
            .url-info {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Video Playback Test</h1>
        <p>Job ID: <code>{job_id}</code></p>
        <p>File: <code>{file_path}</code></p>
        <p>Size: {(file_path.stat().st_size / (1024*1024)):.2f} MB</p>
        
        <div class="url-info">
            <h3>Video URL 1 (FileResponse):</h3>
            <p><a href="{video_url}" target="_blank">{video_url}</a></p>
        </div>
        
        <div class="video-container">
            <h3>Video Player 1:</h3>
            <video controls autoplay muted>
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <div class="url-info">
            <h3>Video URL 2 (StreamingResponse):</h3>
            <p><a href="{stream_url}" target="_blank">{stream_url}</a></p>
        </div>
        
        <div class="video-container">
            <h3>Video Player 2:</h3>
            <video controls autoplay muted>
                <source src="{stream_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <p><a href="/download_video/{job_id}" style="background: #4CAF50; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px;">Download Video</a></p>
        <p><a href="/" style="color: #666;">← Back to main app</a></p>
    </body>
    </html>
    """
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


def _range_response_for_file(file_path: Path, request: Request, chunk_size: int = 8 * 1024):
    """Return a StreamingResponse supporting HTTP Range requests for the given file."""
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise HTTPException(status_code=500, detail="File is empty")

    range_header = request.headers.get("range")

    # FULL FILE RESPONSE (no Range header)
    if range_header is None:
        return FileResponse(
            path=str(file_path),
            media_type="video/mp4",
            filename=file_path.name,
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Content-Type": "video/mp4",   # 🔥 TARAYICI İÇİN ZORUNLU
            },
        )

    # PARTIAL RESPONSE (Has Range)
    try:
        units, range_spec = range_header.split("=", 1)
        if units.strip() != "bytes":
            raise ValueError("Only 'bytes' range supported")

        start_str, end_str = range_spec.split("-", 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1

        if start > end or end >= file_size:
            raise ValueError("Invalid range")
    except Exception:
        return FileResponse(
            path=str(file_path),
            media_type="video/mp4",
            filename=file_path.name,
            headers={"Content-Type": "video/mp4"}
        )

    length = end - start + 1

    def file_iterator(path: Path, start: int, length: int):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Content-Type": "video/mp4",
    }

    return StreamingResponse(
        file_iterator(file_path, start, length),
        status_code=206,
        media_type="video/mp4",
        headers=headers
    )
