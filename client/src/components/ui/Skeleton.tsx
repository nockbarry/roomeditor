const pulse = "animate-pulse bg-gray-800 rounded";

export function SkeletonLine({ className = "" }: { className?: string }) {
  return <div className={`${pulse} h-3 ${className}`} />;
}

export function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`${pulse} ${className}`} />;
}

export function SkeletonSegmentList({ count = 5 }: { count?: number }) {
  return (
    <div className="space-y-1">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex items-center gap-2 px-3 py-2">
          <div className={`${pulse} w-3 h-3 rounded-sm shrink-0`} />
          <div className={`${pulse} h-3 flex-1`} />
          <div className={`${pulse} h-3 w-8`} />
        </div>
      ))}
    </div>
  );
}

export function SkeletonViewer() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 z-10">
      <div className="flex flex-col items-center gap-3">
        <div className={`${pulse} w-12 h-12 rounded-full`} />
        <div className={`${pulse} h-3 w-24`} />
      </div>
    </div>
  );
}

export function SkeletonGrid({ count = 9 }: { count?: number }) {
  return (
    <div className="p-2">
      <div className="grid grid-cols-3 gap-1">
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className={`${pulse} aspect-square rounded`} />
        ))}
      </div>
    </div>
  );
}
