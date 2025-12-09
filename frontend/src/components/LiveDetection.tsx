import React, { useRef, useState, useCallback, useEffect } from "react";
import { Link2, Play, Square, Radio, AlertCircle, Gauge, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DetectResponse } from "@/types/detection";

interface LiveDetectionProps {
  apiBaseUrl: string;
}

export const LiveDetection = ({ apiBaseUrl }: LiveDetectionProps) => {
  const [streamUrl, setStreamUrl] = useState<string>("");
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFps, setCurrentFps] = useState<number>(0);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [latency, setLatency] = useState<number>(0);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const processingRef = useRef<boolean>(false);
  const fpsHistoryRef = useRef<number[]>([]);

  const processFrame = useCallback(async () => {
    if (!processingRef.current || !imgRef.current || !canvasRef.current) return;

    const img = imgRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx) return;

    try {
      // Draw current image to canvas
      canvas.width = img.naturalWidth || 640;
      canvas.height = img.naturalHeight || 480;
      ctx.drawImage(img, 0, 0);

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

      // Calculate metrics
      const frameTime = endTime - startTime;
      setLatency(frameTime);
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
      // Process next frame after a short delay
      setTimeout(() => requestAnimationFrame(processFrame), 100);
    }
  }, [apiBaseUrl]);

  const handleConnect = () => {
    if (!streamUrl) {
      setError("Please enter a stream URL");
      return;
    }

    setError(null);
    setIsConnected(true);
  };

  const handleDisconnect = () => {
    setIsConnected(false);
    processingRef.current = false;
    setIsProcessing(false);
    setCurrentFps(0);
    setDetectionCount(0);
    setLatency(0);
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
      {/* Stream URL Input */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <Link2 className="w-4 h-4 text-primary" />
          Live Stream Connection
        </h3>
        <div className="flex gap-3">
          <Input
            type="url"
            placeholder="Enter stream URL (e.g., http://camera-ip/video.mjpg)"
            value={streamUrl}
            onChange={(e) => setStreamUrl(e.target.value)}
            disabled={isConnected}
            className="flex-1"
          />
          {!isConnected ? (
            <Button onClick={handleConnect} variant="glow">
              <Play className="w-4 h-4" />
              Connect
            </Button>
          ) : (
            <Button onClick={handleDisconnect} variant="destructive">
              <Square className="w-4 h-4" />
              Disconnect
            </Button>
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          Supports MJPEG streams, IP cameras, and HTTP image endpoints
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
          <AlertCircle className="w-4 h-4 text-destructive" />
          <span className="text-sm text-destructive">{error}</span>
        </div>
      )}

      {/* Live Stream Display */}
      {isConnected && (
        <>
          {/* Controls */}
          <div className="flex items-center gap-3">
            <Button
              onClick={isProcessing ? stopProcessing : startProcessing}
              variant={isProcessing ? "destructive" : "glow"}
            >
              {isProcessing ? (
                <>
                  <Square className="w-4 h-4" />
                  Stop Detection
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start Detection
                </>
              )}
            </Button>

            {isProcessing && (
              <div className="flex items-center gap-4 ml-auto">
                <div className="fps-badge">
                  <Gauge className="w-3 h-3" />
                  {currentFps.toFixed(1)} FPS
                </div>
                <span className="text-xs text-muted-foreground">
                  {detectionCount} objects • {latency.toFixed(0)}ms latency
                </span>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Source Stream */}
            <div className="glass-panel p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                  <Radio className="w-4 h-4 text-destructive" />
                  Live Source
                </h3>
                {isConnected && (
                  <div className="flex items-center gap-2">
                    <div className="pulse-dot" />
                    <span className="text-xs text-success">Live</span>
                  </div>
                )}
              </div>
              <div className="canvas-container bg-background">
                <img
                  ref={imgRef}
                  src={streamUrl}
                  alt="Live stream"
                  className="w-full h-full object-contain"
                  crossOrigin="anonymous"
                  onError={() => setError("Failed to load stream. Check the URL and CORS settings.")}
                />
              </div>
            </div>

            {/* Detection Output */}
            <div className="glass-panel p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-primary" />
                  Detection Output
                </h3>
                {isProcessing && (
                  <RefreshCw className="w-4 h-4 text-primary animate-spin" />
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
        </>
      )}

      {/* Placeholder when not connected */}
      {!isConnected && (
        <div className="glass-panel p-12 text-center">
          <Radio className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-foreground mb-2">
            No Stream Connected
          </h3>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">
            Enter a live stream URL above to start real-time object detection.
            Supports MJPEG streams from IP cameras and other video sources.
          </p>
        </div>
      )}
    </div>
  );
};
