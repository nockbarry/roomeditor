import { useState } from "react";
import { Sparkles } from "lucide-react";

export function GeneratePanel() {
  const [prompt, setPrompt] = useState("");

  return (
    <div className="p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">
        Generate Object
      </h3>
      <p className="text-xs text-gray-600 mb-3">
        AI object generation will be available in Phase 3.
      </p>
      <div className="flex gap-2">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder='e.g., "red leather sofa"'
          className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-xs focus:outline-none focus:border-blue-500"
          disabled
        />
        <button
          disabled
          className="bg-purple-600/50 px-3 py-1.5 rounded text-xs flex items-center gap-1 opacity-50"
        >
          <Sparkles className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}
