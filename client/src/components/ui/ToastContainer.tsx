import { useToastStore, type ToastType } from "../../stores/toastStore.ts";
import { X } from "lucide-react";
import { useEffect, useState } from "react";

const BORDER_COLORS: Record<ToastType, string> = {
  success: "border-l-emerald-500",
  error: "border-l-red-500",
  info: "border-l-blue-500",
  warning: "border-l-amber-500",
};

const PROGRESS_COLORS: Record<ToastType, string> = {
  success: "bg-emerald-500",
  error: "bg-red-500",
  info: "bg-blue-500",
  warning: "bg-amber-500",
};

function ToastItem({
  id,
  type,
  message,
  duration,
}: {
  id: string;
  type: ToastType;
  message: string;
  duration: number;
}) {
  const removeToast = useToastStore((s) => s.removeToast);
  const [visible, setVisible] = useState(false);
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    // Trigger enter animation
    requestAnimationFrame(() => setVisible(true));
  }, []);

  useEffect(() => {
    if (duration <= 0) return;
    const start = Date.now();
    const interval = setInterval(() => {
      const elapsed = Date.now() - start;
      const remaining = Math.max(0, 100 - (elapsed / duration) * 100);
      setProgress(remaining);
      if (remaining <= 0) clearInterval(interval);
    }, 50);
    return () => clearInterval(interval);
  }, [duration]);

  return (
    <div
      className={`bg-gray-900 border border-gray-800 border-l-4 ${BORDER_COLORS[type]} rounded-lg shadow-lg px-3 py-2.5 max-w-xs transition-all duration-200 ${
        visible ? "opacity-100 translate-x-0" : "opacity-0 translate-x-4"
      }`}
    >
      <div className="flex items-start gap-2">
        <p className="text-xs text-gray-300 flex-1">{message}</p>
        <button
          onClick={() => removeToast(id)}
          className="text-gray-600 hover:text-gray-400 shrink-0"
        >
          <X className="w-3 h-3" />
        </button>
      </div>
      {duration > 0 && (
        <div className="mt-1.5 h-0.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className={`h-full ${PROGRESS_COLORS[type]} transition-all duration-100 ease-linear`}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  );
}

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((t) => (
        <ToastItem key={t.id} {...t} />
      ))}
    </div>
  );
}
