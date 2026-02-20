import { Routes, Route } from "react-router-dom";
import { Dashboard } from "./pages/Dashboard.tsx";
import { Editor } from "./pages/Editor.tsx";
import { Benchmarks } from "./pages/Benchmarks.tsx";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/project/:projectId" element={<Editor />} />
      <Route path="/benchmarks" element={<Benchmarks />} />
    </Routes>
  );
}
