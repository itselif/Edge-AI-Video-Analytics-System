import { Image, Video, Radio } from "lucide-react";
import { DetectionMode } from "@/types/detection";

interface DetectionModeSelectorProps {
  mode: DetectionMode;
  onModeChange: (mode: DetectionMode) => void;
}

export const DetectionModeSelector = ({ mode, onModeChange }: DetectionModeSelectorProps) => {
  const modes: { id: DetectionMode; label: string; icon: React.ReactNode; description: string }[] = [
    {
      id: "image",
      label: "Image",
      icon: <Image className="w-4 h-4" />,
      description: "Upload and analyze images",
    },
    {
      id: "video",
      label: "Video",
      icon: <Video className="w-4 h-4" />,
      description: "Process video files",
    },
    {
      id: "live",
      label: "Live Stream",
      icon: <Radio className="w-4 h-4" />,
      description: "Real-time detection",
    },
  ];

  return (
    <div className="glass-panel p-2 inline-flex gap-1">
      {modes.map((m) => (
        <button
          key={m.id}
          onClick={() => onModeChange(m.id)}
          data-state={mode === m.id ? "active" : "inactive"}
          className="detection-mode-tab flex items-center gap-2"
        >
          {m.icon}
          <span>{m.label}</span>
        </button>
      ))}
    </div>
  );
};
