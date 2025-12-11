import { Cpu, Zap, Activity, Gauge, MemoryStick } from "lucide-react";
import { MetricsResponse } from "@/types/detection";

interface HeaderProps {
  metrics: MetricsResponse | null;
  currentFps?: number;
  isProcessing?: boolean; 
}

export const Header = ({ metrics, currentFps = 0, isProcessing = false }: HeaderProps) => {
  return (
    <header className="border-b border-border/50 bg-card/30 backdrop-blur-xl sticky top-0 z-50">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-info flex items-center justify-center">
                <Cpu className="w-6 h-6 text-primary-foreground" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-success flex items-center justify-center">
                <Activity className="w-2.5 h-2.5 text-primary-foreground" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                <span className="gradient-text">Edge AI</span>
                <span className="text-foreground ml-2">Vision</span>
              </h1>
              <p className="text-xs text-muted-foreground flex items-center gap-2">
                <Zap className="w-3 h-3 text-warning" />
                YOLO + ONNX • FastAPI • Real-time Analytics
              </p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            {/* FPS Göstergesi - Sadece processing sırasında */}
            {isProcessing && currentFps > 0 && (
              <>
                <div className="hidden md:flex items-center gap-2 bg-gradient-to-r from-primary/10 to-info/10 px-3 py-1.5 rounded-lg">
                  <Gauge className="w-4 h-4 text-primary" />
                  <div className="text-right">
                    <p className="text-xs text-muted-foreground">Processing FPS</p>
                    <p className="text-sm font-bold text-foreground font-mono">
                      {currentFps.toFixed(1)}
                      <span className="text-xs text-muted-foreground ml-1">FPS</span>
                    </p>
                  </div>
                </div>
                <div className="h-8 w-px bg-border" />
              </>
            )}

            {metrics && (
              <>
                {/* GPU Memory */}
                {metrics.gpu_memory_used_mb > 0 && (
                  <>
                    <div className="hidden md:flex items-center gap-2">
                      <MemoryStick className="w-4 h-4 text-warning" />
                      <div className="text-right">
                        <p className="text-xs text-muted-foreground">GPU Memory</p>
                        <p className="text-sm font-medium text-foreground font-mono">
                          {Math.round(metrics.gpu_memory_used_mb)}/{Math.round(metrics.gpu_memory_total_mb)} MB
                        </p>
                      </div>
                    </div>
                    <div className="h-8 w-px bg-border" />
                  </>
                )}

                {/* GPU Utilization */}
                {metrics.gpu_utilization > 0 && (
                  <>
                    <div className="hidden md:flex items-center gap-2">
                      <div className="relative">
                        <div className="w-8 h-8">
                          <svg className="w-8 h-8" viewBox="0 0 36 36">
                            <path
                              d="M18 2.0845
                                a 15.9155 15.9155 0 0 1 0 31.831
                                a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none"
                              stroke="#e5e7eb"
                              strokeWidth="3"
                            />
                            <path
                              d="M18 2.0845
                                a 15.9155 15.9155 0 0 1 0 31.831
                                a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none"
                              stroke="#10b981"
                              strokeWidth="3"
                              strokeDasharray={`${metrics.gpu_utilization}, 100`}
                              strokeLinecap="round"
                            />
                          </svg>
                          <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-foreground">
                            {Math.round(metrics.gpu_utilization)}%
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-muted-foreground">GPU Usage</p>
                        <p className="text-sm font-medium text-foreground">
                          {metrics.gpu_name.split(" ")[0]}
                        </p>
                      </div>
                    </div>
                    <div className="h-8 w-px bg-border" />
                  </>
                )}

                {/* Backend Info */}
                <div className="hidden md:flex items-center gap-2">
                  <div className="text-right">
                    <p className="text-xs text-muted-foreground">Latency</p>
                    <p className="text-sm font-medium text-foreground font-mono">
                      {metrics.avg_latency_ms.toFixed(1)}ms
                    </p>
                  </div>
                </div>
                <div className="h-8 w-px bg-border" />

                {/* Status */}
                <div className="flex items-center gap-2">
                  <div className="pulse-dot" />
                  <span className="text-sm font-medium text-success">Online</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};