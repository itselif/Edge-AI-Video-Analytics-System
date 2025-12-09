import { Activity, Gauge, Clock, Cpu, TrendingUp, Layers } from "lucide-react";
import { MetricsResponse } from "@/types/detection";

interface MetricsPanelProps {
  metrics: MetricsResponse | null;
  loading: boolean;
}

export const MetricsPanel = ({ metrics, loading }: MetricsPanelProps) => {
  const formatMs = (v?: number) => (v === undefined ? "-" : `${v.toFixed(1)}`);
  const formatFps = (v?: number) => (v === undefined ? "-" : v.toFixed(1));

  return (
    <div className="glass-panel p-5 animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          Performance Metrics
        </h3>
        {loading && (
          <span className="text-[10px] text-muted-foreground animate-pulse">
            Updating...
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Avg Latency
            </span>
          </div>
          <div className="metric-value">{formatMs(metrics?.avg_latency_ms)}</div>
          <span className="text-[10px] text-muted-foreground">ms</span>
        </div>

        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Moving Avg
            </span>
          </div>
          <div className="metric-value">{formatMs(metrics?.moving_avg_latency_ms)}</div>
          <span className="text-[10px] text-muted-foreground">ms</span>
        </div>

        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <Gauge className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Throughput
            </span>
          </div>
          <div className="metric-value">{formatFps(metrics?.fps)}</div>
          <span className="text-[10px] text-muted-foreground">FPS</span>
        </div>

        <div className="stat-card">
          <div className="flex items-center gap-2 mb-2">
            <Layers className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Total Requests
            </span>
          </div>
          <div className="metric-value">{metrics?.total_requests ?? 0}</div>
          <span className="text-[10px] text-muted-foreground">req</span>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-border/50">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
            Latency Percentiles
          </span>
        </div>
        <div className="flex gap-2">
          <div className="flex-1 bg-secondary/50 rounded-lg p-2 text-center">
            <div className="text-xs text-muted-foreground mb-1">p50</div>
            <div className="text-sm font-mono font-semibold text-foreground">
              {formatMs(metrics?.p50_latency_ms)}ms
            </div>
          </div>
          <div className="flex-1 bg-secondary/50 rounded-lg p-2 text-center">
            <div className="text-xs text-muted-foreground mb-1">p90</div>
            <div className="text-sm font-mono font-semibold text-foreground">
              {formatMs(metrics?.p90_latency_ms)}ms
            </div>
          </div>
          <div className="flex-1 bg-secondary/50 rounded-lg p-2 text-center">
            <div className="text-xs text-muted-foreground mb-1">p95</div>
            <div className="text-sm font-mono font-semibold text-foreground">
              {formatMs(metrics?.p95_latency_ms)}ms
            </div>
          </div>
        </div>
      </div>

      {metrics?.gpu_name && (
        <div className="mt-4 pt-4 border-t border-border/50">
          <div className="flex items-center gap-2 mb-3">
            <Cpu className="w-3.5 h-3.5 text-primary" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              GPU Status
            </span>
          </div>
          <div className="bg-secondary/50 rounded-lg p-3">
            <p className="text-sm font-medium text-foreground mb-2">{metrics.gpu_name}</p>
            {metrics.gpu_utilization != null && (
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Utilization</span>
                  <span className="text-foreground font-mono">{metrics.gpu_utilization.toFixed(0)}%</span>
                </div>
                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-primary to-info rounded-full transition-all duration-500"
                    style={{ width: `${metrics.gpu_utilization}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
