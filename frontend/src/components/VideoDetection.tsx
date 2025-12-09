import React, { useRef, useEffect, useState, useCallback } from "react";
import { Upload, Play, Pause, Square, Video, AlertCircle, Gauge } from "lucide-react";
import { Button } from "@/components/ui/button";
import { BBox, DetectResponse } from "@/types/detection";

interface VideoDetectionProps {
  apiBaseUrl: string;
}

export const VideoDetection = ({ apiBaseUrl }: VideoDetectionProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFps, setCurrentFps] = useState<number>(0);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [dragOver, setDragOver] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const processingRef = useRef<boolean>(false);
  const lastFrameTimeRef = useRef<number>(0);
  const fpsHistoryRef = useRef<number[]>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    processFile(f);
  };

  const processFile = (f: File | null) => {
    setFile(f);
    setError(null);
    setIsPlaying(false);
    setIsProcessing(false);
    setCurrentFps(0);
    setDetectionCount(0);

    if (f) {
      const url = URL.createObjectURL(f);
      setVideoUrl(url);
    } else {
      setVideoUrl(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0] || null;
    if (f && f.type.startsWith("video/")) {
      processFile(f);
    }
  };

  const processFrame = useCallback(async () => {
    if (!processingRef.current || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx || video.paused || video.ended) {
      processingRef.current = false;
      setIsProcessing(false);
      return;
    }

    // Capture current frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    try {
      // Convert canvas to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((b) => resolve(b!), "image/jpeg", 0.8);
      });

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const startTime = performance.now();
      const res = await fetch(`${apiBaseUrl}/detect`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Request failed`);

      const data: DetectResponse = await res.json();
      const endTime = performance.now();

      // Calculate FPS
      const frameTime = endTime - startTime;
      fpsHistoryRef.current.push(1000 / frameTime);
      if (fpsHistoryRef.current.length > 10) fpsHistoryRef.current.shift();
      const avgFps = fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length;
      setCurrentFps(avgFps);
      setDetectionCount(data.num_detections);

      // Draw detections
      ctx.lineWidth = 3;
      data.detections.forEach((det) => {
        const hue = (det.cls_id * 67) % 360;
        ctx.strokeStyle = `hsl(${hue}, 80%, 55%)`;
        ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;

        const width = det.x2 - det.x1;
        const height = det.y2 - det.y1;
        ctx.strokeRect(det.x1, det.y1, width, height);

        const label = `${det.label ?? det.cls_id} ${det.score.toFixed(2)}`;
        ctx.font = "bold 14px JetBrains Mono, monospace";
        const textMetrics = ctx.measureText(label);
        const textW = textMetrics.width + 12;
        const textH = 22;
        const boxY = Math.max(0, det.y1 - textH - 4);
        ctx.fillRect(det.x1, boxY, textW, textH);
        ctx.fillStyle = "#000000";
        ctx.fillText(label, det.x1 + 6, boxY + textH - 6);
      });
    } catch (err) {
      console.error("Frame processing error:", err);
    }

    if (processingRef.current) {
      requestAnimationFrame(processFrame);
    }
  }, [apiBaseUrl]);

  const handlePlay = () => {
    if (!videoRef.current) return;
    videoRef.current.play();
    setIsPlaying(true);
  };

  const handlePause = () => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    setIsPlaying(false);
  };

  const handleStop = () => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    videoRef.current.currentTime = 0;
    setIsPlaying(false);
    processingRef.current = false;
    setIsProcessing(false);
  };

  const startProcessing = () => {
    processingRef.current = true;
    setIsProcessing(true);
    fpsHistoryRef.current = [];
    processFrame();
  };

  const stopProcessing = () => {
    processingRef.current = false;
    setIsProcessing(false);
  };

  useEffect(() => {
    return () => {
      processingRef.current = false;
    };
  }, []);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Upload Zone */}
      <div
        className={`upload-zone ${dragOver ? "drag-over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-secondary flex items-center justify-center">
            <Upload className="w-8 h-8 text-muted-foreground" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-foreground">
              Drop a video here or click to browse
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Supports MP4, WebM, MOV
            </p>
          </div>
          {file && (
            <div className="flex items-center gap-2 px-4 py-2 bg-secondary rounded-lg">
              <Video className="w-4 h-4 text-primary" />
              <span className="text-sm text-foreground">{file.name}</span>
            </div>
          )}
        </div>
      </div>

      {/* Video Controls */}
      {videoUrl && (
        <div className="flex flex-wrap gap-2">
          <Button onClick={handlePlay} disabled={isPlaying} variant="outline" size="sm">
            <Play className="w-4 h-4" />
            Play
          </Button>
          <Button onClick={handlePause} disabled={!isPlaying} variant="outline" size="sm">
            <Pause className="w-4 h-4" />
            Pause
          </Button>
          <Button onClick={handleStop} variant="outline" size="sm">
            <Square className="w-4 h-4" />
            Stop
          </Button>
          <div className="flex-1" />
          <Button
            onClick={isProcessing ? stopProcessing : startProcessing}
            variant={isProcessing ? "destructive" : "glow"}
            size="sm"
          >
            {isProcessing ? "Stop Detection" : "Start Detection"}
          </Button>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
          <AlertCircle className="w-4 h-4 text-destructive" />
          <span className="text-sm text-destructive">{error}</span>
        </div>
      )}

      {/* Video Display */}
      {videoUrl && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass-panel p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Video className="w-4 h-4 text-muted-foreground" />
                Source Video
              </h3>
              {isPlaying && (
                <div className="pulse-dot" />
              )}
            </div>
            <div className="canvas-container">
              <video
                ref={videoRef}
                src={videoUrl}
                className="w-full h-full object-contain"
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => {
                  setIsPlaying(false);
                  processingRef.current = false;
                  setIsProcessing(false);
                }}
                muted
              />
            </div>
          </div>

          <div className="glass-panel p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Gauge className="w-4 h-4 text-primary" />
                Detection Output
              </h3>
              {isProcessing && (
                <div className="flex items-center gap-3">
                  <span className="fps-badge">
                    {currentFps.toFixed(1)} FPS
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {detectionCount} objects
                  </span>
                </div>
              )}
            </div>
            <div className="canvas-container">
              <canvas ref={canvasRef} className="w-full h-full object-contain" />
              {!isProcessing && (
                <div className="canvas-overlay">
                  <p className="text-sm text-muted-foreground">
                    Start detection to see results
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
