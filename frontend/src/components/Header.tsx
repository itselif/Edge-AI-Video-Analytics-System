import { Cpu, Zap, Activity } from "lucide-react";
import { MetricsResponse } from "@/types/detection";

interface HeaderProps {
  metrics: MetricsResponse | null;
}

export const Header = ({ metrics }: HeaderProps) => {
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

          {metrics && (
            <div className="hidden md:flex items-center gap-6">
              <div className="text-right">
                <p className="text-xs text-muted-foreground">Backend</p>
                <p className="text-sm font-medium text-foreground">{metrics.backend}</p>
              </div>
              <div className="h-8 w-px bg-border" />
              <div className="text-right">
                <p className="text-xs text-muted-foreground">Model</p>
                <p className="text-sm font-medium text-foreground truncate max-w-[200px]">
                  {metrics.model_path.split("/").slice(-1)[0]}
                </p>
              </div>
              <div className="h-8 w-px bg-border" />
              <div className="flex items-center gap-2">
                <div className="pulse-dot" />
                <span className="text-sm font-medium text-success">Online</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
