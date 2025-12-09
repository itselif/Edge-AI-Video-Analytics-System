import React, { useRef, useEffect, useState } from "react";
import { Upload, Play, AlertCircle, CheckCircle, Image as ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { BBox, DetectResponse } from "@/types/detection";

interface ImageDetectionProps {
  apiBaseUrl: string;
}

export const ImageDetection = ({ apiBaseUrl }: ImageDetectionProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [detectResult, setDetectResult] = useState<DetectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const originalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const annotatedCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    processFile(f);
  };

  const processFile = (f: File | null) => {
    setFile(f);
    setDetectResult(null);
    setError(null);

    if (f) {
      const url = URL.createObjectURL(f);
      setOriginalUrl(url);
    } else {
      setOriginalUrl(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0] || null;
    if (f && f.type.startsWith("image/")) {
      processFile(f);
    }
  };

  const handleRunDetection = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${apiBaseUrl}/detect`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const data: DetectResponse = await res.json();
      setDetectResult(data);
    } catch (err: any) {
      setError(err?.message || "Detection request failed.");
      setDetectResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Draw original image
  useEffect(() => {
    if (!originalUrl || !originalCanvasRef.current) return;

    const img = new Image();
    img.src = originalUrl;
    img.onload = () => {
      const canvas = originalCanvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const containerWidth = canvas.parentElement?.clientWidth || 480;
      const scale = Math.min(containerWidth / img.width, 1);
      const w = img.width * scale;
      const h = img.height * scale;

      canvas.width = w;
      canvas.height = h;

      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
    };
  }, [originalUrl]);

  // Draw annotated image
  useEffect(() => {
    if (!originalUrl || !annotatedCanvasRef.current) return;

    const img = new Image();
    img.src = originalUrl;
    img.onload = () => {
      const canvas = annotatedCanvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const containerWidth = canvas.parentElement?.clientWidth || 480;
      const scale = Math.min(containerWidth / img.width, 1);
      const w = img.width * scale;
      const h = img.height * scale;

      canvas.width = w;
      canvas.height = h;

      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);

      if (!detectResult) return;

      ctx.lineWidth = 2;

      detectResult.detections.forEach((det) => {
        const x1 = det.x1 * scale;
        const y1 = det.y1 * scale;
        const x2 = det.x2 * scale;
        const y2 = det.y2 * scale;

        const width = x2 - x1;
        const height = y2 - y1;
        if (width < 5 || height < 5) return;

        const hue = (det.cls_id * 67) % 360;
        ctx.strokeStyle = `hsl(${hue}, 80%, 55%)`;
        ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;

        ctx.strokeRect(x1, y1, width, height);

        const label = `${det.label ?? det.cls_id} ${det.score.toFixed(2)}`;
        ctx.font = "bold 12px JetBrains Mono, monospace";
        const textMetrics = ctx.measureText(label);
        const textW = textMetrics.width + 10;
        const textH = 18;

        const boxY = Math.max(0, y1 - textH - 4);
        ctx.fillRect(x1, boxY, textW, textH);
        ctx.fillStyle = "#000000";
        ctx.fillText(label, x1 + 5, boxY + textH - 5);
      });
    };
  }, [originalUrl, detectResult]);

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
          accept="image/*"
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-secondary flex items-center justify-center">
            <Upload className="w-8 h-8 text-muted-foreground" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-foreground">
              Drop an image here or click to browse
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Supports JPG, PNG, WebP
            </p>
          </div>
          {file && (
            <div className="flex items-center gap-2 px-4 py-2 bg-secondary rounded-lg">
              <ImageIcon className="w-4 h-4 text-primary" />
              <span className="text-sm text-foreground">{file.name}</span>
            </div>
          )}
        </div>
      </div>

      {/* Run Detection Button */}
      <Button
        onClick={handleRunDetection}
        disabled={loading || !file}
        variant="glow"
        size="lg"
        className="w-full"
      >
        {loading ? (
          <>
            <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
            Running Detection...
          </>
        ) : (
          <>
            <Play className="w-4 h-4" />
            Run Detection
          </>
        )}
      </Button>

      {/* Error Message */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
          <AlertCircle className="w-4 h-4 text-destructive" />
          <span className="text-sm text-destructive">{error}</span>
        </div>
      )}

      {/* Detection Result Info */}
      {detectResult && (
        <div className="flex items-center gap-4 p-4 bg-success/10 border border-success/20 rounded-lg">
          <CheckCircle className="w-5 h-5 text-success" />
          <div className="flex-1">
            <p className="text-sm font-medium text-foreground">Detection Complete</p>
            <p className="text-xs text-muted-foreground">
              {detectResult.num_detections} objects detected in {detectResult.inference_time_ms.toFixed(2)}ms
            </p>
          </div>
          <div className="fps-badge">
            {(1000 / detectResult.inference_time_ms).toFixed(1)} FPS
          </div>
        </div>
      )}

      {/* Canvas Display */}
      {originalUrl && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass-panel p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <ImageIcon className="w-4 h-4 text-muted-foreground" />
              Original
            </h3>
            <div className="canvas-container flex items-center justify-center">
              <canvas ref={originalCanvasRef} className="max-w-full" />
            </div>
          </div>

          <div className="glass-panel p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-primary" />
              Detections
            </h3>
            <div className="canvas-container flex items-center justify-center">
              {detectResult ? (
                <canvas ref={annotatedCanvasRef} className="max-w-full" />
              ) : (
                <div className="canvas-overlay">
                  <p className="text-sm text-muted-foreground">
                    Run detection to see results
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
