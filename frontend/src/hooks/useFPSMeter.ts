import { useRef, useCallback, useState } from 'react';

export const useFPSMeter = (windowSize: number = 10) => {
  const timestampsRef = useRef<number[]>([]);
  const [fps, setFps] = useState<number>(0);

  const tick = useCallback(() => {
    const now = performance.now();
    timestampsRef.current.push(now);

    // Keep only last N timestamps
    while (timestampsRef.current.length > windowSize) {
      timestampsRef.current.shift();
    }

    // compute fps
    const len = timestampsRef.current.length;
    if (len < 2) {
      setFps(0);
      return;
    }

    const first = timestampsRef.current[0];
    const last = timestampsRef.current[len - 1];
    const elapsed = last - first;
    if (elapsed <= 0) {
      setFps(0);
      return;
    }

    const newFps = ((len - 1) / elapsed) * 1000;
    setFps(newFps);
  }, [windowSize]);

  const reset = useCallback(() => {
    timestampsRef.current = [];
    setFps(0);
  }, []);

  return { tick, reset, fps };
};