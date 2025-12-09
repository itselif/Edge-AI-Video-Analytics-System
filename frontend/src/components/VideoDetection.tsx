import React, { useRef, useState } from "react";
import {
  Upload,
  Play,
  Pause,
  Square,
  Video,
  AlertCircle,
  Gauge,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface VideoDetectionProps {
  apiBaseUrl: string;
}

export const VideoDetection = ({ apiBaseUrl }: VideoDetectionProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [sourceVideoUrl, setSourceVideoUrl] = useState<string | null>(null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(
    null
  );

  const [isPlaying, setIsPlaying] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const sourceVideoRef = useRef<HTMLVideoElement | null>(null);
  const processedVideoRef = useRef<HTMLVideoElement | null>(null);

  // -----------------------------------------------------------
  // File handling
  // -----------------------------------------------------------
  const resetStateForNewFile = () => {
    setProcessedVideoUrl(null);
    setIsPlaying(false);
    setIsProcessing(false);
    setError(null);
  };

  const processFile = (f: File | null) => {
    setFile(f);
    resetStateForNewFile();

    if (f) {
      const url = URL.createObjectURL(f);
      setSourceVideoUrl(url);
    } else {
      setSourceVideoUrl(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    if (f && !f.type.startsWith("video/")) {
      setError("Please select a valid video file.");
      return;
    }
    processFile(f);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0] || null;
    if (!f) return;
    if (!f.type.startsWith("video/")) {
      setError("Please drop a valid video file.");
      return;
    }
    processFile(f);
  };

  // -----------------------------------------------------------
  // Backend call: /detect_video  (offline processing)
  // -----------------------------------------------------------
  const handleStartDetection = async () => {
    if (!file) {
      setError("Please upload a video first.");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setProcessedVideoUrl(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${apiBaseUrl}/detect_video`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Video detection failed with status ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setProcessedVideoUrl(url);
    } catch (err: any) {
      console.error("Video detection error:", err);
      setError(err?.message || "Video detection failed.");
    } finally {
      setIsProcessing(false);
    }
  };

  // -----------------------------------------------------------
  // Playback controls (play/pause/stop both videos in sync)
  // -----------------------------------------------------------
  const handlePlay = () => {
    const src = sourceVideoRef.current;
    const out = processedVideoRef.current;

    if (!src) return;

    // keep processed video roughly in sync with source
    if (out && processedVideoUrl) {
      out.currentTime = src.currentTime;
      void out.play();
    }

    void src.play();
    setIsPlaying(true);
  };

  const handlePause = () => {
    const src = sourceVideoRef.current;
    const out = processedVideoRef.current;

    src && src.pause();
    out && out.pause();
    setIsPlaying(false);
  };

  const handleStop = () => {
    const src = sourceVideoRef.current;
    const out = processedVideoRef.current;

    if (src) {
      src.pause();
      src.currentTime = 0;
    }
    if (out) {
      out.pause();
      out.currentTime = 0;
    }

    setIsPlaying(false);
  };

  // -----------------------------------------------------------

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

      {/* Controls */}
      {sourceVideoUrl && (
        <div className="flex flex-wrap gap-2">
          <Button
            onClick={handlePlay}
            disabled={isPlaying || isProcessing}
            variant="outline"
            size="sm"
          >
            <Play className="w-4 h-4" />
            Play
          </Button>
          <Button
            onClick={handlePause}
            disabled={!isPlaying}
            variant="outline"
            size="sm"
          >
            <Pause className="w-4 h-4" />
            Pause
          </Button>
          <Button onClick={handleStop} variant="outline" size="sm">
            <Square className="w-4 h-4" />
            Stop
          </Button>

          <div className="flex-1" />

          <Button
            onClick={handleStartDetection}
            variant="glow"
            size="sm"
            disabled={isProcessing || !file}
          >
            {isProcessing ? "Processing…" : "Start Detection"}
          </Button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
          <AlertCircle className="w-4 h-4 text-destructive" />
          <span className="text-sm text-destructive">{error}</span>
        </div>
      )}

      {/* Video Panels */}
      {sourceVideoUrl && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Source video */}
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
                ref={sourceVideoRef}
                src={sourceVideoUrl}
                className="w-full h-full object-contain"
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
                muted
                controls={false}
              />
            </div>
          </div>

          {/* Processed video */}
          <div className="glass-panel p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Gauge className="w-4 h-4 text-primary" />
                Detection Output
              </h3>
              {processedVideoUrl && !isProcessing && (
                <span className="fps-badge">Ready</span>
              )}
              {isProcessing && (
                <span className="fps-badge">Processing…</span>
              )}
            </div>
            <div className="canvas-container relative">
              {processedVideoUrl && !isProcessing ? (
                <video
                  ref={processedVideoRef}
                  src={processedVideoUrl}
                  className="w-full h-full object-contain"
                  muted
                  controls
                />
              ) : (
                <>
                  <div className="canvas-overlay">
                    <p className="text-sm text-muted-foreground">
                      {isProcessing
                        ? "Processing video… this may take a moment."
                        : "Start detection to see results"}
                    </p>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
