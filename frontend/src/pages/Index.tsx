import React, { useEffect, useState } from "react";
import { Header } from "@/components/Header";
import { MetricsPanel } from "@/components/MetricsPanel";
import { DetectionModeSelector } from "@/components/DetectionModeSelector";
import { ImageDetection } from "@/components/ImageDetection";
import { VideoDetection } from "@/components/VideoDetection";
import { LiveDetection } from "@/components/LiveDetection";
import { DetectionMode, MetricsResponse } from "@/types/detection";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const Index: React.FC = () => {
  const [mode, setMode] = useState<DetectionMode>("image");
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  // Fetch metrics periodically
  useEffect(() => {
    const fetchMetrics = async () => {
      setLoadingMetrics(true);
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (!res.ok) throw new Error(`Metrics request failed`);
        const data: MetricsResponse = await res.json();
        setMetrics(data);
      } catch {
        // Ignore errors, keep previous metrics
      } finally {
        setLoadingMetrics(false);
      }
    };

    fetchMetrics();
    const id = setInterval(fetchMetrics, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Background glow effect */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background: "radial-gradient(ellipse at 50% 0%, hsl(174 72% 50% / 0.08), transparent 50%)",
        }}
      />

      <Header metrics={metrics} />

      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr,320px] gap-6">
          {/* Main Content */}
          <div className="space-y-6">
            {/* Mode Selector */}
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h2 className="text-2xl font-bold text-foreground">Object Detection</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  Select a detection mode and upload your media
                </p>
              </div>
              <DetectionModeSelector mode={mode} onModeChange={setMode} />
            </div>

            {/* Detection Content */}
            <div className="min-h-[500px]">
              {mode === "image" && <ImageDetection apiBaseUrl={API_BASE_URL} />}
              {mode === "video" && <VideoDetection apiBaseUrl={API_BASE_URL} />}
              {mode === "live" && <LiveDetection apiBaseUrl={API_BASE_URL} />}
            </div>
          </div>

          {/* Sidebar */}
          <aside className="space-y-6">
            <MetricsPanel metrics={metrics} loading={loadingMetrics} />

            {/* Quick Stats */}
            <div className="glass-panel p-5">
              <h3 className="text-sm font-semibold mb-4">Detection Info</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Mode</span>
                  <span className="text-foreground capitalize">{mode}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Backend</span>
                  <span className="text-foreground">{metrics?.backend || "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <span className="text-success flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-success" />
                    Ready
                  </span>
                </div>
              </div>
            </div>

            {/* Help Card */}
            <div className="glass-panel p-5 gradient-border">
              <h3 className="text-sm font-semibold mb-2">Quick Tips</h3>
              <ul className="text-xs text-muted-foreground space-y-2">
                <li>• Use high-quality images for better detection accuracy</li>
                <li>• For video, processing speed depends on resolution</li>
                <li>• Live streams require CORS headers enabled</li>
                <li>• GPU acceleration significantly improves performance</li>
              </ul>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
};

export default Index;
