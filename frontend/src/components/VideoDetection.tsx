import React, { useRef, useEffect, useState, useCallback } from "react";
import { Upload, Play, Pause, Square, Video, AlertCircle, Gauge, Download, ExternalLink, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { BBox, DetectResponse } from "@/types/detection";
import { useFPSMeter } from '@/hooks/useFPSMeter';

interface VideoDetectionProps {
  apiBaseUrl: string;
}

type ProcessingMode = "realtime" | "async";

export const VideoDetection = ({ apiBaseUrl }: VideoDetectionProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoLoadError, setVideoLoadError] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [dragOver, setDragOver] = useState(false);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>("async");
  
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>("");
  const [progress, setProgress] = useState<number | null>(null);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [processedVideoFps, setProcessedVideoFps] = useState<number>(0);
  const [processedDetectionCount, setProcessedDetectionCount] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const processedVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const processingRef = useRef<boolean>(false);
  const pollRef = useRef<number | null>(null);

  const { tick, reset, fps: currentFps } = useFPSMeter(10);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    processFile(f);
  };

  const processFile = (f: File | null) => {
    if (processedVideoUrl?.startsWith('blob:')) {
      URL.revokeObjectURL(processedVideoUrl);
    }
    
    setFile(f);
    setError(null);
    setIsPlaying(false);
    setIsProcessing(false);
    setDetectionCount(0);
    setProcessedDetectionCount(0);
    setProcessedVideoUrl(null);
    setJobId(null);
    setJobStatus("");
    setProgress(null);
    setEtaSeconds(null);
    setVideoLoadError(false);
    setProcessedVideoFps(0);
    reset();

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

  const fetchVideoAsBlob = async (url: string): Promise<string | null> => {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to fetch video: ${response.status}`);
      
      const blob = await response.blob();
      const contentType = response.headers.get('content-type') || '';
      const mimeType = contentType.includes('webm') || url.includes('.webm') ? 'video/webm' : 'video/mp4';
      
      const fixedBlob = new Blob([blob], { type: mimeType });
      return URL.createObjectURL(fixedBlob);
    } catch (err) {
      console.error("Failed to fetch video as blob:", err);
      return null;
    }
  };

  const startAsyncProcessing = async () => {
    if (!file) {
      setError("Please upload a video first.");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setProcessedVideoUrl(null);
    setProgress(0);
    setJobStatus("uploading");
    setProcessedVideoFps(0);
    setProcessedDetectionCount(0);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${apiBaseUrl}/detect_video_async`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed with status ${res.status}`);
      }

      const data = await res.json();
      const jid = data.job_id;

      setJobId(jid);
      setJobStatus("queued");

      pollRef.current = window.setInterval(async () => {
        try {
          const sres = await fetch(`${apiBaseUrl}/detect_video_status/${jid}`);
          if (!sres.ok) throw new Error("Status check failed");
          
          const sdata = await sres.json();
          setJobStatus(sdata.status);
          setProgress(sdata.progress ?? null);
          setEtaSeconds(sdata.eta_seconds ?? null);

          // Get detection count from backend if available
          if (sdata.fps !== undefined) {
            setProcessedVideoFps(sdata.fps);
            
          }         
          if (sdata.detection_count) {
            setProcessedDetectionCount(sdata.detection_count);
          }

          if (sdata.status === "done") {
            if (pollRef.current) {
              clearInterval(pollRef.current);
              pollRef.current = null;
            }

            setIsVideoLoading(true);
            const videoUrl = `${apiBaseUrl}/stream_video/${jid}`;
            const blobUrl = await fetchVideoAsBlob(videoUrl);
            
            if (blobUrl) {
              setProcessedVideoUrl(blobUrl);
            } else {
              setProcessedVideoUrl(`${apiBaseUrl}/processed/${jid}?t=${Date.now()}`);
            }
            
            setIsProcessing(false);
          }

          if (sdata.status === "failed") {
            if (pollRef.current) {
              clearInterval(pollRef.current);
              pollRef.current = null;
            }
            setError(sdata.error || "Processing failed");
            setIsProcessing(false);
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 1000);
    } catch (err: any) {
      setError(err?.message || "Video detection failed.");
      setIsProcessing(false);
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

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    try {
      tick();

      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((b) => resolve(b!), "image/jpeg", 0.8);
      });

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const res = await fetch(`${apiBaseUrl}/detect`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Request failed`);

      const data: DetectResponse = await res.json();
      setDetectionCount(data.num_detections);

      ctx.lineWidth = 2;
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
  }, [apiBaseUrl, tick]);

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

  const startRealtimeProcessing = () => {
    processingRef.current = true;
    setIsProcessing(true);
    reset();
    tick();
    processFrame();
  };

  const stopRealtimeProcessing = () => {
    processingRef.current = false;
    setIsProcessing(false);
  };

  const handleStartDetection = () => {
    if (processingMode === "async") {
      startAsyncProcessing();
    } else {
      if (isProcessing) {
        stopRealtimeProcessing();
      } else {
        startRealtimeProcessing();
      }
    }
  };

  const tryAlternativeSource = async (type: 'stream' | 'direct') => {
    if (!jobId) return;
    
    setIsVideoLoading(true);
    setError(null);
    
    const url = type === 'stream' 
      ? `${apiBaseUrl}/stream_video/${jobId}`
      : `${apiBaseUrl}/processed/${jobId}`;
    
    const blobUrl = await fetchVideoAsBlob(url);
    
    if (blobUrl) {
      if (processedVideoUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(processedVideoUrl);
      }
      setProcessedVideoUrl(blobUrl);
    } else {
      setError("Failed to load video. Try downloading instead.");
      setIsVideoLoading(false);
    }
  };

  const formatEta = (seconds: number | null) => {
    if (seconds === null) return "";
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${minutes}m ${secs}s`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "queued": case "uploading": return "text-warning";
      case "processing": return "text-info";
      case "done": return "text-success";
      case "failed": return "text-destructive";
      default: return "text-muted-foreground";
    }
  };

  const handleProcessedVideoLoad = () => {
    // We rely on backend-reported FPS (polled via job status). Do not override it here.
    // Keep hook to clear loading flag when video metadata is ready.
    if (processedVideoRef.current) {
      const video = processedVideoRef.current;
      if (video.readyState >= 1) {
        setIsVideoLoading(false);
      }
    }
  };

  useEffect(() => {
    return () => {
      processingRef.current = false;
      if (pollRef.current) clearInterval(pollRef.current);
      if (processedVideoUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(processedVideoUrl);
      }
    };
  }, [processedVideoUrl]);

  return (
    <div className="space-y-6 animate-fade-in">
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
              <span className="text-xs text-muted-foreground">
                ({Math.round(file.size / 1024 / 1024)} MB)
              </span>
            </div>
          )}
        </div>
      </div>

      {videoUrl && (
        <div className="flex gap-2 p-1 bg-secondary rounded-lg w-fit">
          <button
            onClick={() => setProcessingMode("async")}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              processingMode === "async"
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Full Video Processing
          </button>
          <button
            onClick={() => setProcessingMode("realtime")}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              processingMode === "realtime"
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Real-time Detection
          </button>
        </div>
      )}

      {processingMode === "async" && jobStatus && (
        <div className="flex items-center justify-between p-3 glass-panel rounded-lg">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Status:</span>
            <span className={`text-sm font-semibold ${getStatusColor(jobStatus)}`}>
              {jobStatus.toUpperCase()}
            </span>
            {isVideoLoading && jobStatus === "done" && (
              <span className="text-xs text-info animate-pulse ml-2">Loading video...</span>
            )}
          </div>
          {jobStatus === "done" && videoLoadError && (
            <Button
              onClick={() => tryAlternativeSource('stream')}
              size="sm"
              variant="ghost"
              className="h-8"
            >
              <RefreshCw className="w-3 h-3 mr-1" /> Retry
            </Button>
          )}
        </div>
      )}

      {processingMode === "async" && isProcessing && progress !== null && (
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium">
              Processing: {Math.round(progress)}%
            </span>
            {etaSeconds !== null && (
              <span className="text-sm text-muted-foreground">
                ETA: {formatEta(etaSeconds)}
              </span>
            )}
          </div>
          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
            <div
              className="bg-primary h-full transition-all duration-300"
              style={{ width: `${Math.min(progress, 100)}%` }}
            />
          </div>
        </div>
      )}

      {videoUrl && (
        <div className="flex flex-wrap gap-2">
          {processingMode === "realtime" && (
            <>
              <Button onClick={handlePlay} disabled={isPlaying} variant="outline" size="sm">
                <Play className="w-4 h-4 mr-1" />
                Play
              </Button>
              <Button onClick={handlePause} disabled={!isPlaying} variant="outline" size="sm">
                <Pause className="w-4 h-4 mr-1" />
                Pause
              </Button>
              <Button onClick={handleStop} variant="outline" size="sm">
                <Square className="w-4 h-4 mr-1" />
                Stop
              </Button>
            </>
          )}
          <div className="flex-1" />
          <Button
            onClick={handleStartDetection}
            disabled={isProcessing && processingMode === "async"}
            variant={isProcessing && processingMode === "realtime" ? "destructive" : "glow"}
            size="sm"
          >
            {isProcessing
              ? processingMode === "realtime"
                ? "Stop Detection"
                : "Processing..."
              : "Start Detection"}
          </Button>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
          <AlertCircle className="w-4 h-4 text-destructive" />
          <span className="text-sm text-destructive">{error}</span>
        </div>
      )}

      {videoUrl && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass-panel p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Video className="w-4 h-4 text-muted-foreground" />
                Source Video
              </h3>
              {isPlaying && <div className="pulse-dot" />}
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
                controls={processingMode === "async"}
              />
            </div>
          </div>

          <div className="glass-panel p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Gauge className="w-4 h-4 text-primary" />
                Detection Output
              </h3>
              <div className="flex items-center gap-3">
                {processingMode === "realtime" && isProcessing && (
                  <>
                    <span className="fps-badge">{currentFps.toFixed(1)} FPS</span>
                    <span className="text-xs text-muted-foreground">{detectionCount} objects</span>
                  </>
                )}
                {processingMode === "async" && jobId && (
                  <>
                    <span className="fps-badge">{processedVideoFps > 0 ? processedVideoFps.toFixed(1) : "30.0"} FPS</span>
                    <span className="text-xs text-muted-foreground">
                      {processedDetectionCount > 0 ? processedDetectionCount : "0"} objects
                    </span>
                    {!isVideoLoading && <div className="pulse-dot" />}
                  </>
                )}
              </div>
            </div>

            <div className="canvas-container relative">
              {processingMode === "realtime" ? (
                <>
                  <canvas ref={canvasRef} className="w-full h-full object-contain" />
                  {!isProcessing && (
                    <div className="canvas-overlay">
                      <p className="text-sm text-muted-foreground">Start detection to see results</p>
                    </div>
                  )}
                </>
              ) : (
                <>
                  {jobStatus === "done" && processedVideoUrl ? (
                    <>
                      {isVideoLoading && (
                        <div className="absolute inset-0 bg-background/80 flex items-center justify-center z-10 rounded-lg">
                          <div className="text-center">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2" />
                            <p className="text-sm text-muted-foreground">Loading video...</p>
                          </div>
                        </div>
                      )}
                      <video
                        ref={processedVideoRef}
                        key={processedVideoUrl}
                        src={processedVideoUrl}
                        className="w-full h-full object-contain"
                        controls
                        preload="auto"
                        playsInline
                        onLoadedData={handleProcessedVideoLoad}
                        onLoadedMetadata={handleProcessedVideoLoad}
                        onCanPlay={() => {
                          setIsVideoLoading(false);
                          handleProcessedVideoLoad();
                        }}
                        onError={(e) => {
                          console.error("Video playback error:", e);
                          setIsVideoLoading(false);
                          setVideoLoadError(true);
                          setError("Video playback failed. Try alternative sources below.");
                        }}
                      />
                      <div className="absolute bottom-2 right-2 flex gap-1">
                        <button
                          onClick={() => tryAlternativeSource('stream')}
                          className="bg-primary/80 hover:bg-primary text-primary-foreground px-2 py-1 rounded text-xs"
                        >
                          Stream
                        </button>
                        <button
                          onClick={() => tryAlternativeSource('direct')}
                          className="bg-secondary hover:bg-secondary/80 text-secondary-foreground px-2 py-1 rounded text-xs"
                        >
                          Direct
                        </button>
                      </div>
                    </>
                  ) : (
                    <div className="canvas-overlay">
                      {isProcessing ? (
                        <div className="text-center">
                          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2" />
                          <p className="text-sm text-muted-foreground">Processing video...</p>
                          {progress !== null && (
                            <p className="text-xs text-muted-foreground mt-1">{Math.round(progress)}% complete</p>
                          )}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">Start detection to process video</p>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>

            {processingMode === "async" && jobStatus === "done" && jobId && (
              <div className="mt-3 flex gap-2 flex-wrap">
                <a
                  href={`${apiBaseUrl}/processed/${jobId}`}
                  download={`processed_${jobId}.mp4`}
                  className="inline-flex items-center gap-1 text-xs bg-primary text-primary-foreground px-3 py-1.5 rounded-lg hover:bg-primary/90 transition-colors"
                >
                  <Download className="w-3 h-3" />
                  Download
                </a>
                <a
                  href={`${apiBaseUrl}/test_video/${jobId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs bg-secondary text-secondary-foreground px-3 py-1.5 rounded-lg hover:bg-secondary/80 transition-colors"
                >
                  <ExternalLink className="w-3 h-3" />
                  Test Page
                </a>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};