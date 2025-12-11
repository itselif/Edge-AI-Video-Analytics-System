import React, { useRef, useState, useCallback, useEffect } from "react";
import { Link2, Play, Square, Radio, AlertCircle, Gauge, RefreshCw, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DetectResponse } from "@/types/detection";

interface LiveDetectionProps {
  apiBaseUrl: string;
}

export const LiveDetection = ({ apiBaseUrl }: LiveDetectionProps) => {
  const [streamUrl, setStreamUrl] = useState<string>("");
  const [sourceType, setSourceType] = useState<"url" | "camera">("url");
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFps, setCurrentFps] = useState<number>(0);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [latency, setLatency] = useState<number>(0);
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>("");
  const [isCameraReady, setIsCameraReady] = useState(false);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const processingRef = useRef<boolean>(false);
  const fpsHistoryRef = useRef<number[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const lastDetectionsRef = useRef<DetectResponse["detections"]>([]);

  // Get available camera devices
  useEffect(() => {
    const getCameras = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setCameraDevices(videoDevices);
        if (videoDevices.length > 0) {
          setSelectedCamera(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.error("Failed to list cameras:", err);
      }
    };

    getCameras();
  }, []);

  const startCamera = async (deviceId?: string) => {
    try {
      // Stop existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints = {
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play()
          .then(() => {
            setIsCameraReady(true);
            setError(null);
          })
          .catch(err => {
            console.error("Video play failed:", err);
            setError("Failed to play camera stream");
          });
      }
    } catch (err) {
      console.error("Camera access error:", err);
      setError("Camera access denied or not available");
      setIsCameraReady(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.pause();
    }
    
    setIsCameraReady(false);
  };

  const processFrame = useCallback(async () => {
    if (!processingRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Choose source element
    const sourceEl = sourceType === "camera" ? videoRef.current : imgRef.current;
    if (!sourceEl) return;

    // Camera video check
    if (sourceType === "camera") {
      const video = videoRef.current;
      if (!video || video.readyState < 2 || video.videoWidth === 0) {
        setTimeout(() => requestAnimationFrame(processFrame), 100);
        return;
      }
    }

    try {
      // Get dimensions
      let width, height;
      if (sourceType === "camera") {
        const video = videoRef.current!;
        width = video.videoWidth;
        height = video.videoHeight;
      } else {
        const img = imgRef.current!;
        width = img.naturalWidth || 640;
        height = img.naturalHeight || 480;
      }

      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(sourceEl as CanvasImageSource, 0, 0, width, height);

      // Convert to blob
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

      if (!res.ok) throw new Error(`Detection request failed`);

      const data: DetectResponse = await res.json();
      const endTime = performance.now();

      // Update metrics
      const frameTime = endTime - startTime;
      setLatency(frameTime);
      
      fpsHistoryRef.current.push(1000 / frameTime);
      if (fpsHistoryRef.current.length > 10) fpsHistoryRef.current.shift();
      const avgFps = fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length;
      setCurrentFps(avgFps);
      setDetectionCount(data.num_detections);

      // Store latest detections for continuous rendering
      lastDetectionsRef.current = data.detections;

    } catch (err) {
      console.error("Frame processing error:", err);
    }

    if (processingRef.current) {
      setTimeout(() => requestAnimationFrame(processFrame), 100);
    }
  }, [apiBaseUrl, sourceType]);

  // Continuous rendering loop — shows cached detections even between requests
  useEffect(() => {
    if (!isProcessing || !canvasRef.current) return;

    let rafId: number;
    
    const renderLoop = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx) {
        rafId = requestAnimationFrame(renderLoop);
        return;
      }

      const sourceEl = sourceType === "camera" ? videoRef.current : imgRef.current;
      if (!sourceEl) {
        rafId = requestAnimationFrame(renderLoop);
        return;
      }

      // Get proper dimensions
      let width = canvas.width;
      let height = canvas.height;
      
      if (sourceType === "camera") {
        const video = videoRef.current;
        if (video && video.readyState >= 2) {
          width = video.videoWidth;
          height = video.videoHeight;
        }
      } else {
        const img = imgRef.current;
        if (img && img.naturalWidth) {
          width = img.naturalWidth;
          height = img.naturalHeight;
        }
      }

      // Update canvas size if needed
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      // Draw video frame
      try {
        ctx.drawImage(sourceEl as CanvasImageSource, 0, 0, width, height);
      } catch (e) {
        // Silently fail, will retry next frame
      }

      // Draw cached detections continuously
      if (lastDetectionsRef.current && lastDetectionsRef.current.length > 0) {
        ctx.lineWidth = 3;
        ctx.font = "bold 16px Arial, sans-serif";

        lastDetectionsRef.current.forEach((det) => {
          const hue = (det.cls_id * 67) % 360;
          ctx.strokeStyle = `hsl(${hue}, 80%, 55%)`;
          ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;

          const boxWidth = det.x2 - det.x1;
          const boxHeight = det.y2 - det.y1;
          ctx.strokeRect(det.x1, det.y1, boxWidth, boxHeight);

          const label = `${det.label ?? det.cls_id} ${det.score.toFixed(2)}`;
          const textMetrics = ctx.measureText(label);
          const textWidth = textMetrics.width + 12;
          const textHeight = 24;
          const boxY = Math.max(0, det.y1 - textHeight - 4);

          ctx.fillRect(det.x1, boxY, textWidth, textHeight);
          ctx.fillStyle = "#FFFFFF";
          ctx.fillText(label, det.x1 + 6, boxY + textHeight - 6);
        });
      }

      rafId = requestAnimationFrame(renderLoop);
    };

    rafId = requestAnimationFrame(renderLoop);

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [isProcessing, sourceType]);

  const handleConnect = () => {
    setError(null);
    
    if (sourceType === "camera") {
      startCamera(selectedCamera)
        .then(() => {
          setIsConnected(true);
        })
        .catch(err => {
          setError("Failed to start camera: " + err.message);
        });
      return;
    }

    if (!streamUrl.trim()) {
      setError("Please enter a stream URL");
      return;
    }

    setIsConnected(true);
  };

  const handleDisconnect = () => {
    setIsConnected(false);
    setIsProcessing(false);
    processingRef.current = false;
    setCurrentFps(0);
    setDetectionCount(0);
    setLatency(0);
    
    if (sourceType === "camera") {
      stopCamera();
    }
  };

  const startProcessing = () => {
    if (sourceType === "camera" && !isCameraReady) {
      setError("Camera is not ready yet");
      return;
    }
    
    processingRef.current = true;
    setIsProcessing(true);
    fpsHistoryRef.current = [];
    processFrame();
  };

  const stopProcessing = () => {
    processingRef.current = false;
    setIsProcessing(false);
  };

  // Handle camera device change
  useEffect(() => {
    if (sourceType === "camera" && isConnected && selectedCamera) {
      startCamera(selectedCamera);
    }
  }, [selectedCamera, sourceType, isConnected]);

  // Cleanup
  useEffect(() => {
    return () => {
      processingRef.current = false;
      stopCamera();
    };
  }, []);

  return (
    <div className="space-y-6">
      {/* Connection Settings */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <Link2 className="w-4 h-4 text-primary" />
          Live Stream Source
        </h3>
        
        <div className="space-y-4">
          {/* Source Type Selection */}
          <div className="flex gap-4">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSourceType("url")}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  sourceType === "url"
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
                }`}
              >
                Stream URL
              </button>
              <button
                onClick={() => setSourceType("camera")}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  sourceType === "camera"
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
                }`}
              >
                Web Camera
              </button>
            </div>
          </div>

          {/* Camera Selection */}
          {sourceType === "camera" && cameraDevices.length > 0 && (
            <div className="flex items-center gap-3">
              <label className="text-sm font-medium">Select Camera:</label>
              <select
                value={selectedCamera}
                onChange={(e) => setSelectedCamera(e.target.value)}
                className="flex-1 bg-background border border-input rounded-md px-3 py-2 text-sm"
                disabled={isConnected}
              >
                {cameraDevices.map((device) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label || `Camera ${device.deviceId.slice(0, 8)}`}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* URL Input */}
          {sourceType === "url" && (
            <div className="flex gap-3">
              <Input
                type="url"
                placeholder="Enter stream URL (e.g., http://192.168.1.100/video.mjpg)"
                value={streamUrl}
                onChange={(e) => setStreamUrl(e.target.value)}
                disabled={isConnected}
                className="flex-1"
              />
            </div>
          )}

          {/* Connect/Disconnect Button */}
          <div className="flex gap-3">
            {!isConnected ? (
              <Button onClick={handleConnect} variant="glow" className="flex items-center gap-2">
                <Play className="w-4 h-4" />
                Connect
              </Button>
            ) : (
              <Button onClick={handleDisconnect} variant="destructive" className="flex items-center gap-2">
                <Square className="w-4 h-4" />
                Disconnect
              </Button>
            )}
          </div>
        </div>

        <p className="text-xs text-muted-foreground mt-3">
          {sourceType === "url" 
            ? "Supports MJPEG streams, IP cameras, and HTTP image endpoints" 
            : "Requires camera permissions. Make sure your camera is connected and accessible."}
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
          {/* Detection Controls */}
          <div className="flex items-center justify-between">
            <Button
              onClick={isProcessing ? stopProcessing : startProcessing}
              variant={isProcessing ? "destructive" : "glow"}
              className="flex items-center gap-2"
              disabled={sourceType === "camera" && !isCameraReady}
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
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-primary/10 rounded-full">
                  <Gauge className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">{currentFps.toFixed(1)} FPS</span>
                </div>
                <div className="text-sm text-muted-foreground">
                  <span className="font-medium">{detectionCount}</span> objects detected
                  <span className="mx-2">•</span>
                  <span>{latency.toFixed(0)}ms latency</span>
                </div>
              </div>
            )}
          </div>

          {/* Video Display Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Live Source */}
            <div className="glass-panel p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                  <Radio className="w-4 h-4 text-blue-500" />
                  Live Source
                </h3>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-xs text-green-600">Live</span>
                </div>
              </div>
              
              <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                {sourceType === "camera" ? (
                  <video
                    ref={videoRef}
                    className="w-full h-full object-contain"
                    playsInline
                    muted
                    autoPlay
                    onLoadedData={() => setIsCameraReady(true)}
                    onError={() => setError("Failed to load camera feed")}
                  />
                ) : (
                  <img
                    ref={imgRef}
                    src={streamUrl}
                    alt="Live stream"
                    className="w-full h-full object-contain"
                    crossOrigin="anonymous"
                    onLoad={() => setIsCameraReady(true)}
                    onError={() => setError("Failed to load stream. Check URL and CORS.")}
                  />
                )}
                
                {sourceType === "camera" && !isCameraReady && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2" />
                      <p className="text-sm text-white">Initializing camera...</p>
                    </div>
                  </div>
                )}
              </div>
              
              <div className="mt-2 text-xs text-muted-foreground">
                {sourceType === "camera" 
                  ? "Live camera feed" 
                  : `Streaming from: ${streamUrl}`}
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
              
              <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                <canvas
                  ref={canvasRef}
                  className="w-full h-full object-contain"
                />
                
                {!isProcessing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/70">
                    <div className="text-center p-4">
                      <Video className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                      <p className="text-lg font-medium text-white mb-1">Detection Output</p>
                      <p className="text-sm text-gray-300">
                        Start detection to see real-time object recognition results
                      </p>
                    </div>
                  </div>
                )}
                
                {isProcessing && detectionCount === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <div className="text-center">
                      <RefreshCw className="w-8 h-8 text-white animate-spin mx-auto mb-2" />
                      <p className="text-sm text-white">Processing frames...</p>
                    </div>
                  </div>
                )}
              </div>
              
              <div className="mt-2 text-xs text-muted-foreground">
                {isProcessing 
                  ? `Detecting ${detectionCount} objects in real-time`
                  : "Detection results will appear here"}
              </div>
            </div>
          </div>
        </>
      )}

      {/* Connection Placeholder */}
      {!isConnected && (
        <div className="glass-panel p-12 text-center">
          <div className="w-16 h-16 bg-secondary rounded-full flex items-center justify-center mx-auto mb-4">
            <Radio className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">
            No Active Connection
          </h3>
          <p className="text-sm text-muted-foreground max-w-md mx-auto mb-6">
            {sourceType === "camera"
              ? "Connect to your web camera to start real-time object detection"
              : "Enter a live stream URL or select camera to begin real-time detection"}
          </p>
          
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button
              onClick={handleConnect}
              variant="outline"
              className="flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              Connect Now
            </Button>
            
            {sourceType === "camera" && cameraDevices.length === 0 && (
              <div className="text-xs text-amber-600">
                No cameras detected. Please check your device connections.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
