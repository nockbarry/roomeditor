import { useEffect, useCallback, useMemo, useRef } from "react";
import * as THREE from "three";
import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import { useSegmentStore } from "../../stores/segmentStore.ts";
import { useEditorStore } from "../../stores/editorStore.ts";
import { useCollabStore } from "../../stores/collabStore.ts";
import { useSplatScene } from "../../hooks/useSplatScene.ts";
import { useSplatHighlight } from "../../hooks/useSplatHighlight.ts";
import { useBrushTool } from "../../hooks/useBrushTool.ts";
import { useEraserActions } from "../../hooks/useEraserTool.ts";
import { CropOverlay } from "./CropOverlay.tsx";
import { ModelProvenance } from "./ModelProvenance.tsx";
import { Box, Loader2, Camera } from "lucide-react";

interface SplatViewerProps {
  projectId: string;
}

function formatGaussianCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

export function SplatViewer({ projectId }: SplatViewerProps) {
  const {
    containerRef, loading, error, numSplats, sceneFormat, loadPly, loadScene,
    splatMeshRef, positionsRef, sceneRef, cameraRef, rendererRef,
    kdTreeRef, getHitPointRef,
    onPickRef, onHoverRef, onGizmoDragEndRef, updateGizmo,
  } = useSplatScene();

  const plyUrl = useAnySplatStore((s) => s.plyUrl);
  const plyVersion = useAnySplatStore((s) => s.plyVersion);
  const isRunning = useAnySplatStore((s) => s.isRunning);
  const lastRun = useAnySplatStore((s) => s.lastRun);
  const activeFormat = useAnySplatStore((s) => s.activeFormat);
  const showComparison = useAnySplatStore((s) => s.showComparison);
  const comparisonBeforeUrl = useAnySplatStore((s) => s.comparisonBeforeUrl);
  const setShowComparison = useAnySplatStore((s) => s.setShowComparison);

  const segmentIndexMap = useSegmentStore((s) => s.segmentIndexMap);
  const segments = useSegmentStore((s) => s.segments);
  const selectedSegmentIds = useSegmentStore((s) => s.selectedSegmentIds);
  const selectSegment = useSegmentStore((s) => s.selectSegment);
  const batchTransform = useSegmentStore((s) => s.batchTransform);
  const setHoveredSegment = useSegmentStore((s) => s.setHoveredSegment);
  const hoveredSegmentId = useSegmentStore((s) => s.hoveredSegmentId);
  const toolMode = useEditorStore((s) => s.toolMode);
  const cameraMode = useEditorStore((s) => s.cameraMode);
  const loadSceneInMemory = useEditorStore((s) => s.loadScene);
  const sceneLoaded = useEditorStore((s) => s.sceneLoaded);
  const collabUsers = useCollabStore((s) => s.users);

  // Load scene into server memory when the viewer loads a scene
  useEffect(() => {
    if (numSplats > 0 && !sceneLoaded && plyUrl?.includes(`/data/${projectId}/`)) {
      loadSceneInMemory(projectId);
    }
  }, [numSplats, sceneLoaded, plyUrl, projectId, loadSceneInMemory]);

  // Wire brush tool for brush/eraser modes
  const getHitPoint = useCallback(
    (event: MouseEvent) => getHitPointRef.current?.(event) ?? null,
    [getHitPointRef],
  );
  useBrushTool(containerRef, kdTreeRef, positionsRef, getHitPoint);

  // Wire eraser finalization (soft-delete brush selection on mouseup)
  const { finalizeErase } = useEraserActions(projectId);
  useEffect(() => {
    if (toolMode !== "eraser") return;
    const onUp = () => finalizeErase();
    window.addEventListener("mouseup", onUp);
    return () => window.removeEventListener("mouseup", onUp);
  }, [toolMode, finalizeErase]);

  // Crop tool state from editor store
  const cropBox = useEditorStore((s) => s.cropBox);
  const cropSphere = useEditorStore((s) => s.cropSphere);
  const setCropBox = useEditorStore((s) => s.setCropBox);
  const setCropSphere = useEditorStore((s) => s.setCropSphere);

  // Initialize crop volumes when switching to crop tools
  useEffect(() => {
    if (toolMode === "crop-box" && !cropBox && positionsRef.current) {
      const pos = positionsRef.current;
      const n = pos.length / 3;
      if (n === 0) return;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      const step = Math.max(1, Math.floor(n / 1000));
      for (let i = 0; i < n; i += step) {
        const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      }
      const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
      const sx = (maxX - minX) * 0.25, sy = (maxY - minY) * 0.25, sz = (maxZ - minZ) * 0.25;
      setCropBox({ min: [cx - sx, cy - sy, cz - sz], max: [cx + sx, cy + sy, cz + sz] });
    } else if (toolMode === "crop-sphere" && !cropSphere && positionsRef.current) {
      const pos = positionsRef.current;
      const n = pos.length / 3;
      if (n === 0) return;
      let sx = 0, sy = 0, sz = 0, count = 0;
      const step = Math.max(1, Math.floor(n / 1000));
      for (let i = 0; i < n; i += step) {
        sx += pos[i * 3]; sy += pos[i * 3 + 1]; sz += pos[i * 3 + 2]; count++;
      }
      const cx = sx / count, cy = sy / count, cz = sz / count;
      setCropSphere({ center: [cx, cy, cz], radius: 0.5 });
    } else if (toolMode !== "crop-box" && toolMode !== "crop-sphere") {
      setCropBox(null);
      setCropSphere(null);
    }
  }, [toolMode, cropBox, cropSphere, positionsRef, setCropBox, setCropSphere]);

  // Render remote cursor spheres in the 3D scene
  const cursorMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    const existing = cursorMeshesRef.current;
    const activeIds = new Set<string>();

    for (const user of collabUsers) {
      if (!user.cursor) continue;
      activeIds.add(user.user_id);

      let mesh = existing.get(user.user_id);
      if (!mesh) {
        const geo = new THREE.SphereGeometry(0.03, 8, 8);
        const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
        mesh = new THREE.Mesh(geo, mat);
        mesh.renderOrder = 999;
        scene.add(mesh);
        existing.set(user.user_id, mesh);
      }

      // Update position and color
      mesh.position.set(user.cursor[0], user.cursor[1], user.cursor[2]);
      (mesh.material as THREE.MeshBasicMaterial).color.set(user.color);
    }

    // Remove cursors for users who left or no longer have cursor data
    for (const [uid, mesh] of existing) {
      if (!activeIds.has(uid)) {
        scene.remove(mesh);
        mesh.geometry.dispose();
        (mesh.material as THREE.MeshBasicMaterial).dispose();
        existing.delete(uid);
      }
    }
  }, [collabUsers, sceneRef]);

  // Cleanup cursor meshes on unmount
  useEffect(() => {
    return () => {
      const scene = sceneRef.current;
      for (const [, mesh] of cursorMeshesRef.current) {
        if (scene) scene.remove(mesh);
        mesh.geometry.dispose();
        (mesh.material as THREE.MeshBasicMaterial).dispose();
      }
      cursorMeshesRef.current.clear();
    };
  }, [sceneRef]);

  // Wire pick callback → segment selection + send cursor to collab
  const sendCursorMove = useCollabStore((s) => s.sendCursorMove);
  useEffect(() => {
    onPickRef.current = (splatIdx, point) => {
      // Send cursor position to collaborators
      if (point) {
        sendCursorMove([point.x, point.y, point.z]);
      }

      if (splatIdx === null || !segmentIndexMap) {
        selectSegment(null);
        return;
      }
      const segValue = segmentIndexMap[splatIdx]; // 0=unassigned, 1+=segment
      if (segValue === 0) {
        selectSegment(null);
        return;
      }
      const seg = segments[segValue - 1];
      selectSegment(seg?.id ?? null);
    };
  }, [segmentIndexMap, segments, selectSegment, sendCursorMove, onPickRef]);

  // Wire hover callback → segment hover highlight
  useEffect(() => {
    onHoverRef.current = (splatIdx) => {
      if (splatIdx === null || !segmentIndexMap) {
        setHoveredSegment(null);
        return;
      }
      const segValue = segmentIndexMap[splatIdx];
      if (segValue === 0) {
        setHoveredSegment(null);
        return;
      }
      const seg = segments[segValue - 1];
      setHoveredSegment(seg?.id ?? null);
    };
  }, [segmentIndexMap, segments, setHoveredSegment, onHoverRef]);

  // Highlight selected segments
  useSplatHighlight(splatMeshRef);

  // Wire gizmo drag end → batch transform
  useEffect(() => {
    onGizmoDragEndRef.current = (delta) => {
      if (selectedSegmentIds.length === 0) return;
      const transform: { translation?: number[]; rotation?: number[]; scale?: number[] } = {};
      if (delta.translation) transform.translation = delta.translation;
      if (delta.rotation) transform.rotation = delta.rotation;
      if (delta.scale) transform.scale = delta.scale;
      batchTransform(projectId, transform).then(() => {
        useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
      });
    };
  }, [selectedSegmentIds, batchTransform, projectId, onGizmoDragEndRef]);

  // Compute centroid of selected segments for gizmo placement
  const selectedCentroid = useMemo(() => {
    if (selectedSegmentIds.length === 0 || !positionsRef.current) return null;
    const positions = positionsRef.current;

    // Collect all gaussian IDs from selected segments
    const selectedSegs = segments.filter((s) => selectedSegmentIds.includes(s.id));
    if (selectedSegs.every((s) => s.n_gaussians === 0)) return null;

    // Compute approximate centroid from segment data
    // Use positions array and segment index map
    if (!segmentIndexMap) return null;

    let sx = 0, sy = 0, sz = 0, count = 0;
    const selectedValues = new Set<number>();
    for (const seg of selectedSegs) {
      const listIdx = segments.indexOf(seg);
      if (listIdx >= 0) selectedValues.add(listIdx + 1);
    }

    const step = Math.max(1, Math.floor(segmentIndexMap.length / 10000));
    for (let i = 0; i < segmentIndexMap.length; i += step) {
      if (selectedValues.has(segmentIndexMap[i])) {
        sx += positions[i * 3];
        sy += positions[i * 3 + 1];
        sz += positions[i * 3 + 2];
        count++;
      }
    }

    if (count === 0) return null;
    return new THREE.Vector3(sx / count, sy / count, sz / count);
  }, [selectedSegmentIds, segments, segmentIndexMap, positionsRef]);

  // Update gizmo when selection or tool mode changes
  useEffect(() => {
    updateGizmo(selectedCentroid, toolMode);
  }, [selectedCentroid, toolMode, updateGizmo]);

  useEffect(() => {
    if (!plyUrl) return;
    const t = Date.now();
    const isScenePly = plyUrl.endsWith("scene.ply");
    const isSpzUrl = plyUrl.endsWith(".spz");

    if (isSpzUrl) {
      // Directly loading an SPZ file (e.g. from format toggle)
      const posUrl = plyUrl.replace(/\.spz$/, ".positions.bin");
      const plyFallback = plyUrl.replace(/\.spz$/, ".ply");
      console.log(`[SplatViewer] Loading SPZ directly: ${plyUrl}`);
      loadScene(`${plyUrl}?t=${t}`, `${posUrl}?t=${t}`, `${plyFallback}?t=${t}`);
    } else if (isScenePly && activeFormat === "spz") {
      // scene.ply URL + SPZ preferred — try SPZ with PLY fallback
      const spzUrl = plyUrl.replace(/\.ply$/, ".spz");
      const posUrl = plyUrl.replace(/\.ply$/, ".positions.bin");
      console.log(`[SplatViewer] Loading via loadScene: SPZ=${spzUrl}, PLY fallback=${plyUrl}`);
      loadScene(`${spzUrl}?t=${t}`, `${posUrl}?t=${t}`, `${plyUrl}?t=${t}`);
    } else if (plyUrl.endsWith(".ply")) {
      // PLY format preferred or non-scene PLY (stage files, comparisons)
      const spzUrl = plyUrl.replace(/\.ply$/, ".spz");
      if (spzUrl !== plyUrl) {
        // Has a potential positions sidecar — use loadScene with PLY as primary
        const posUrl = plyUrl.replace(/\.ply$/, ".positions.bin");
        console.log(`[SplatViewer] Loading PLY directly: ${plyUrl}`);
        loadScene(`${plyUrl}?t=${t}`, `${posUrl}?t=${t}`);
      } else {
        console.log(`[SplatViewer] Loading via loadPly: ${plyUrl}`);
        loadPly(`${plyUrl}?t=${t}`);
      }
    } else {
      console.log(`[SplatViewer] Loading via loadPly: ${plyUrl}`);
      loadPly(`${plyUrl}?t=${t}`);
    }
  }, [plyUrl, plyVersion, activeFormat, loadPly, loadScene]);

  const takeScreenshot = useCallback(() => {
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    if (!renderer || !scene || !camera) return;
    // Render one frame to ensure the canvas is up-to-date
    renderer.render(scene, camera);
    const dataUrl = renderer.domElement.toDataURL("image/png");
    const link = document.createElement("a");
    link.download = `screenshot-${Date.now()}.png`;
    link.href = dataUrl;
    link.click();
  }, [rendererRef, sceneRef, cameraRef]);

  return (
    <div className={`flex-1 min-h-0 relative rounded-lg overflow-hidden bg-gray-900 border border-gray-800 ${hoveredSegmentId !== null ? "cursor-pointer" : ""}`}>
      {/* Canvas container — always rendered so useSplatScene can initialize */}
      <div ref={containerRef} className="absolute inset-0" />

      {/* Crop volume overlay */}
      <CropOverlay
        scene={sceneRef.current}
        cropBox={cropBox}
        cropSphere={cropSphere}
      />

      {/* Placeholder overlay — shown when no PLY and not running */}
      {!plyUrl && !isRunning && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 z-10">
          <div className="text-center">
            <Box className="w-10 h-10 mx-auto mb-3 text-gray-700" />
            <p className="text-sm text-gray-500">No result yet</p>
            <p className="text-xs text-gray-600 mt-1">Select frames and click Rebuild</p>
          </div>
        </div>
      )}

      {/* Loading/running overlay */}
      {(loading || isRunning) && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-10 pointer-events-none">
          <div className="text-center">
            <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2 text-emerald-400" />
            <p className="text-xs text-gray-400">
              {isRunning
                ? "Running AnySplat..."
                : numSplats > 0
                  ? `Loading... ${formatGaussianCount(numSplats)} splats`
                  : "Loading 3D scene..."}
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute top-2 left-2 right-2 bg-red-900/80 text-red-200 text-xs p-2 rounded z-10">
          Failed to load: {error}
        </div>
      )}

      {!loading && (numSplats > 0 || lastRun) && (
        <div className="absolute bottom-2 left-2 flex items-center gap-1.5 z-10">
          <ModelProvenance projectId={projectId} sceneFormat={sceneFormat} />
          <span className="text-xs text-gray-500 bg-gray-900/80 px-2 py-1 rounded">
            {formatGaussianCount(numSplats || lastRun?.n_gaussians || 0)} Gaussians
          </span>
        </div>
      )}

      {/* Before/After comparison toggle */}
      {showComparison && comparisonBeforeUrl && (
        <div className="absolute top-2 right-2 z-10 flex flex-col gap-1">
          <div className="flex bg-gray-900/90 rounded-lg p-0.5 border border-gray-700/50 shadow-lg">
            <button
              onClick={() => {
                useAnySplatStore.setState((s) => ({
                  plyUrl: comparisonBeforeUrl,
                  plyVersion: s.plyVersion + 1,
                }));
              }}
              className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                plyUrl === comparisonBeforeUrl
                  ? "bg-amber-600 text-white"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              Before
            </button>
            <button
              onClick={() => {
                useAnySplatStore.setState((s) => ({
                  plyUrl: `/data/${projectId}/scene.ply`,
                  plyVersion: s.plyVersion + 1,
                }));
              }}
              className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                plyUrl !== comparisonBeforeUrl
                  ? "bg-emerald-600 text-white"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              After
            </button>
          </div>
          <button
            onClick={() => {
              setShowComparison(false);
              useAnySplatStore.setState((s) => ({
                plyUrl: `/data/${projectId}/scene.ply`,
                plyVersion: s.plyVersion + 1,
                comparisonBeforeUrl: null,
              }));
            }}
            className="text-[9px] text-gray-500 hover:text-white text-center"
          >
            Close Compare
          </button>
        </div>
      )}

      <div className="absolute bottom-2 right-2 flex items-center gap-1.5 z-10">
        {numSplats > 0 && (
          <button
            onClick={takeScreenshot}
            className="flex items-center gap-1 text-[10px] text-gray-400 hover:text-white bg-gray-900/80 hover:bg-gray-800/90 px-2 py-1 rounded transition-colors"
            title="Take screenshot"
          >
            <Camera className="w-3 h-3" />
          </button>
        )}
        <span className="text-[10px] text-gray-600 bg-gray-900/80 px-2 py-1 rounded">
          {cameraMode === "fps" ? (
            <>
              <span className="text-blue-400 font-medium">Fly</span>
              {" "}&middot; WASD move &middot; QE up/down &middot; LMB look &middot; Scroll speed &middot; V orbit
            </>
          ) : (
            <>
              <span className="text-gray-400 font-medium">Orbit</span>
              {" "}&middot; LMB orbit &middot; Scroll zoom &middot; RMB pan &middot; V fly
            </>
          )}
        </span>
      </div>
    </div>
  );
}
