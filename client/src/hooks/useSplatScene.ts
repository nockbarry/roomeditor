import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { buildKDTree, queryNearest, type KDTreeNode } from "./useKDTree.ts";
import type { ToolMode } from "../types/scene.ts";

/** Max f_rest properties Spark supports (SH degree 3 = 45). */
const MAX_F_REST = 45;

/** Drag threshold in pixels — clicks below this trigger pick, above are orbits. */
const PICK_DRAG_THRESHOLD = 6;

/** Minimum interval between hover computations (ms). */
const HOVER_THROTTLE_MS = 50;

/**
 * Patch a PLY ArrayBuffer so Spark can load it.
 * Spark only supports 0/9/24/45 f_rest properties. If the PLY has more
 * (e.g. 72 for SH degree 4), we rename the excess in the header to _pad_N.
 * This preserves the vertex stride so binary data stays untouched.
 */
function patchPlyHeader(buf: ArrayBuffer): ArrayBuffer {
  const headerSearch = new Uint8Array(buf, 0, Math.min(buf.byteLength, 65536));
  const decoder = new TextDecoder("ascii");
  const headerText = decoder.decode(headerSearch);
  const endIdx = headerText.indexOf("end_header\n");
  if (endIdx === -1) return buf;

  const headerEnd = endIdx + "end_header\n".length;
  const header = headerText.slice(0, headerEnd);

  const fRestRegex = /property\s+float\s+f_rest_(\d+)/g;
  let maxFRest = -1;
  let match;
  while ((match = fRestRegex.exec(header)) !== null) {
    const idx = parseInt(match[1], 10);
    if (idx > maxFRest) maxFRest = idx;
  }

  const numFRest = maxFRest + 1;
  if (numFRest <= MAX_F_REST) return buf;

  let patched = header;
  for (let i = MAX_F_REST; i < numFRest; i++) {
    patched = patched.replace(
      `property float f_rest_${i}`,
      `property float _pad_${i - MAX_F_REST}`,
    );
  }

  const encoder = new TextEncoder();
  const newHeaderBytes = encoder.encode(patched);
  const binaryData = new Uint8Array(buf, headerEnd);
  const result = new Uint8Array(newHeaderBytes.length + binaryData.length);
  result.set(newHeaderBytes, 0);
  result.set(binaryData, newHeaderBytes.length);
  return result.buffer;
}

/**
 * Parse PLY header and extract ALL xyz positions into a Float32Array.
 * Also computes centroid and bounding radius (sampled).
 */
function extractPlyPositions(buf: ArrayBuffer): {
  positions: Float32Array;
  nVerts: number;
  centroid: THREE.Vector3;
  radius: number;
} | null {
  const headerSearch = new Uint8Array(buf, 0, Math.min(buf.byteLength, 65536));
  const headerText = new TextDecoder("ascii").decode(headerSearch);
  const endIdx = headerText.indexOf("end_header\n");
  if (endIdx === -1) return null;
  const headerEnd = endIdx + "end_header\n".length;
  const header = headerText.slice(0, headerEnd);

  const vertMatch = header.match(/element vertex (\d+)/);
  if (!vertMatch) return null;
  const nVerts = parseInt(vertMatch[1], 10);
  if (nVerts === 0) return null;

  // Parse property list to find byte offset and stride
  const lines = header.split("\n");
  let byteOffset = 0;
  let xOff = -1,
    yOff = -1,
    zOff = -1;
  let inVertex = false;

  for (const line of lines) {
    if (line.startsWith("element vertex")) {
      inVertex = true;
      byteOffset = 0;
      continue;
    }
    if (line.startsWith("element ") && inVertex) break;
    if (!inVertex) continue;

    const propMatch = line.match(/property\s+(\w+)\s+(\w+)/);
    if (!propMatch) continue;
    const [, type, name] = propMatch;
    const size =
      type === "double" ? 8 : type === "uchar" || type === "uint8" ? 1 : 4;

    if (name === "x") xOff = byteOffset;
    else if (name === "y") yOff = byteOffset;
    else if (name === "z") zOff = byteOffset;
    byteOffset += size;
  }
  const stride = byteOffset;

  if (xOff < 0 || yOff < 0 || zOff < 0 || stride === 0) return null;

  // Extract ALL positions
  const data = new DataView(buf, headerEnd);
  const positions = new Float32Array(nVerts * 3);

  for (let i = 0; i < nVerts; i++) {
    const base = i * stride;
    positions[i * 3] = data.getFloat32(base + xOff, true);
    positions[i * 3 + 1] = data.getFloat32(base + yOff, true);
    positions[i * 3 + 2] = data.getFloat32(base + zOff, true);
  }

  // Compute centroid + radius from sampled positions
  const sampleStep = Math.max(1, Math.floor(nVerts / 2000));
  let sx = 0,
    sy = 0,
    sz = 0,
    count = 0;
  for (let i = 0; i < nVerts; i += sampleStep) {
    sx += positions[i * 3];
    sy += positions[i * 3 + 1];
    sz += positions[i * 3 + 2];
    count++;
  }
  const centroid = new THREE.Vector3(sx / count, sy / count, sz / count);

  let maxDistSq = 0;
  for (let i = 0; i < nVerts; i += sampleStep) {
    const dx = positions[i * 3] - centroid.x;
    const dy = positions[i * 3 + 1] - centroid.y;
    const dz = positions[i * 3 + 2] - centroid.z;
    const dSq = dx * dx + dy * dy + dz * dz;
    if (dSq > maxDistSq) maxDistSq = dSq;
  }

  return { positions, nVerts, centroid, radius: Math.sqrt(maxDistSq) };
}

/**
 * Parse a positions sidecar binary file (.positions.bin).
 * Format: uint32 N, float32 cx cy cz, float32 radius, then N*3 float32 positions.
 */
function parsePositionsSidecar(buf: ArrayBuffer): {
  positions: Float32Array;
  nVerts: number;
  centroid: THREE.Vector3;
  radius: number;
} | null {
  if (buf.byteLength < 20) return null;
  const view = new DataView(buf);
  const nVerts = view.getUint32(0, true);
  const cx = view.getFloat32(4, true);
  const cy = view.getFloat32(8, true);
  const cz = view.getFloat32(12, true);
  const radius = view.getFloat32(16, true);
  if (buf.byteLength < 20 + nVerts * 12) return null;
  const positions = new Float32Array(buf, 20, nVerts * 3);
  return { positions, nVerts, centroid: new THREE.Vector3(cx, cy, cz), radius };
}

export interface GizmoDelta {
  translation?: [number, number, number];
  rotation?: [number, number, number];
  scale?: [number, number, number];
}

export interface SceneFormatInfo {
  format: "spz" | "ply";
  sizeMB: number;
}

export interface SplatSceneHandle {
  containerRef: React.RefObject<HTMLDivElement | null>;
  loading: boolean;
  error: string | null;
  numSplats: number;
  sceneFormat: SceneFormatInfo | null;
  loadPly: (url: string) => void;
  loadScene: (sceneUrl: string, positionsUrl: string, plyFallbackUrl?: string) => void;
  splatMeshRef: React.RefObject<SplatMesh | null>;
  positionsRef: React.RefObject<Float32Array | null>;
  sceneRef: React.RefObject<THREE.Scene | null>;
  cameraRef: React.RefObject<THREE.PerspectiveCamera | null>;
  rendererRef: React.RefObject<THREE.WebGLRenderer | null>;
  onPickRef: React.RefObject<
    ((splatIdx: number | null, point: THREE.Vector3 | null) => void) | null
  >;
  onHoverRef: React.RefObject<((splatIdx: number | null) => void) | null>;
  onGizmoDragEndRef: React.RefObject<((delta: GizmoDelta) => void) | null>;
  updateGizmo: (centroid: THREE.Vector3 | null, mode: ToolMode) => void;
}

export function useSplatScene(): SplatSceneHandle {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const sparkRef = useRef<SparkRenderer | null>(null);
  const frameGroupRef = useRef<THREE.Group | null>(null);
  const splatMeshRef = useRef<SplatMesh | null>(null);
  const positionsRef = useRef<Float32Array | null>(null);
  const kdTreeRef = useRef<KDTreeNode | null>(null);
  const onPickRef = useRef<
    ((splatIdx: number | null, point: THREE.Vector3 | null) => void) | null
  >(null);
  const onHoverRef = useRef<((splatIdx: number | null) => void) | null>(null);
  const onGizmoDragEndRef = useRef<((delta: GizmoDelta) => void) | null>(null);
  const transformControlsRef = useRef<TransformControls | null>(null);
  const proxyMeshRef = useRef<THREE.Mesh | null>(null);
  const proxyStartPosRef = useRef<THREE.Vector3 | null>(null);
  const proxyStartQuatRef = useRef<THREE.Quaternion | null>(null);
  const proxyStartScaleRef = useRef<THREE.Vector3 | null>(null);
  const initedRef = useRef(false);
  const loadIdRef = useRef(0);

  // For drag detection
  const mouseDownPosRef = useRef<{ x: number; y: number } | null>(null);

  // For hover throttle
  const lastHoverTimeRef = useRef(0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [numSplats, setNumSplats] = useState(0);
  const [sceneFormat, setSceneFormat] = useState<SceneFormatInfo | null>(null);

  // Initialize Three.js + Spark once when container mounts
  useEffect(() => {
    const container = containerRef.current;
    if (!container || initedRef.current) return;

    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;

    initedRef.current = true;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000);
    camera.position.set(-1, 2, -2);
    cameraRef.current = camera;

    // Orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(-3, 1, 8);
    controls.update();
    controlsRef.current = controls;

    // SparkRenderer — add to scene so it can render splats
    const spark = new SparkRenderer({ renderer });
    scene.add(spark);
    sparkRef.current = spark;

    // Frame group with Y-down convention (quaternion 180deg around X)
    const frame = new THREE.Group();
    frame.quaternion.set(1, 0, 0, 0);
    scene.add(frame);
    frameGroupRef.current = frame;

    // Raycaster shared by dblclick, pick, and hover
    const raycaster = new THREE.Raycaster();

    // Helper: transform a world-space hit point to mesh local space
    const worldToLocal = (hitWorld: THREE.Vector3): void => {
      const frameGroup = frameGroupRef.current;
      if (frameGroup) {
        const invQ = frameGroup.quaternion.clone().invert();
        hitWorld.applyQuaternion(invQ);
      }
    };

    // Double-click to re-center orbit target
    const handleDblClick = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
      const ray = raycaster.ray;

      const viewDir = new THREE.Vector3()
        .subVectors(controls.target, camera.position)
        .normalize();
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
        viewDir,
        controls.target,
      );
      const hit = new THREE.Vector3();
      if (ray.intersectPlane(plane, hit)) {
        controls.target.copy(hit);
        controls.update();
      }
    };
    container.addEventListener("dblclick", handleDblClick);

    // --- Click-to-pick with drag detection ---
    const handleMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return; // left button only
      mouseDownPosRef.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (e.button !== 0) return;
      const downPos = mouseDownPosRef.current;
      mouseDownPosRef.current = null;
      if (!downPos) return;

      // Check drag distance
      const dx = e.clientX - downPos.x;
      const dy = e.clientY - downPos.y;
      if (Math.sqrt(dx * dx + dy * dy) > PICK_DRAG_THRESHOLD) return;

      // This was a click, not a drag — do picking
      const mesh = splatMeshRef.current;
      const positions = positionsRef.current;
      const kdTree = kdTreeRef.current;
      const onPick = onPickRef.current;
      if (!mesh || !positions || !kdTree || !onPick) return;

      const rect = container.getBoundingClientRect();
      const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);

      const intersects: {
        distance: number;
        point: THREE.Vector3;
        object: THREE.Object3D;
      }[] = [];
      mesh.raycast(raycaster, intersects);

      if (intersects.length === 0) {
        onPick(null, null);
        return;
      }

      // Transform hit point from world space to mesh local (undo frame group transform)
      const hitWorld = intersects[0].point.clone();
      worldToLocal(hitWorld);

      const nearestIdx = queryNearest(
        kdTree,
        positions,
        hitWorld.x,
        hitWorld.y,
        hitWorld.z,
      );

      if (nearestIdx >= 0) {
        onPick(nearestIdx, intersects[0].point);
      } else {
        onPick(null, null);
      }
    };

    container.addEventListener("mousedown", handleMouseDown);
    container.addEventListener("mouseup", handleMouseUp);

    // --- Hover handler with throttle ---
    const handleMouseMove = (e: MouseEvent) => {
      const now = performance.now();
      if (now - lastHoverTimeRef.current < HOVER_THROTTLE_MS) return;
      lastHoverTimeRef.current = now;

      const mesh = splatMeshRef.current;
      const positions = positionsRef.current;
      const kdTree = kdTreeRef.current;
      const onHover = onHoverRef.current;
      if (!mesh || !positions || !kdTree || !onHover) return;

      const rect = container.getBoundingClientRect();
      const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);

      const intersects: {
        distance: number;
        point: THREE.Vector3;
        object: THREE.Object3D;
      }[] = [];
      mesh.raycast(raycaster, intersects);

      if (intersects.length === 0) {
        onHover(null);
        return;
      }

      const hitWorld = intersects[0].point.clone();
      worldToLocal(hitWorld);

      const nearestIdx = queryNearest(
        kdTree,
        positions,
        hitWorld.x,
        hitWorld.y,
        hitWorld.z,
      );

      onHover(nearestIdx >= 0 ? nearestIdx : null);
    };

    const handleMouseLeave = () => {
      const onHover = onHoverRef.current;
      if (onHover) onHover(null);
    };

    container.addEventListener("mousemove", handleMouseMove);
    container.addEventListener("mouseleave", handleMouseLeave);

    // Animation loop
    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    // Resize observer
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        }
      }
    });
    ro.observe(container);

    return () => {
      container.removeEventListener("dblclick", handleDblClick);
      container.removeEventListener("mousedown", handleMouseDown);
      container.removeEventListener("mouseup", handleMouseUp);
      container.removeEventListener("mousemove", handleMouseMove);
      container.removeEventListener("mouseleave", handleMouseLeave);
      ro.disconnect();
      renderer.setAnimationLoop(null);
      if (splatMeshRef.current) {
        splatMeshRef.current.dispose();
        splatMeshRef.current = null;
      }
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement) {
        renderer.domElement.parentElement.removeChild(renderer.domElement);
      }
      initedRef.current = false;
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
      sparkRef.current = null;
      frameGroupRef.current = null;
    };
  }, []);

  const loadPly = useCallback((url: string) => {
    const frame = frameGroupRef.current;
    if (!frame) return;

    // Bump load ID so any in-flight loads become stale
    const thisLoadId = ++loadIdRef.current;

    // Dispose old splat mesh immediately
    if (splatMeshRef.current) {
      frame.remove(splatMeshRef.current);
      splatMeshRef.current.dispose();
      splatMeshRef.current = null;
    }
    positionsRef.current = null;
    kdTreeRef.current = null;

    setLoading(true);
    setError(null);
    setNumSplats(0);
    setSceneFormat(null);

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.arrayBuffer();
      })
      .then((buf) => {
        // If a newer load was started, discard this one
        if (loadIdRef.current !== thisLoadId) return;

        setSceneFormat({ format: "ply", sizeMB: Math.round(buf.byteLength / 1048576) });
        const patched = patchPlyHeader(buf);

        // Extract positions for picking + compute bounds for camera
        const extracted = extractPlyPositions(patched);
        if (extracted) {
          positionsRef.current = extracted.positions;

          // Build k-d tree for fast nearest-neighbor lookups
          const t0 = performance.now();
          kdTreeRef.current = buildKDTree(extracted.positions);
          console.log(
            `[useSplatScene] K-D tree built in ${(performance.now() - t0).toFixed(0)}ms for ${extracted.nVerts.toLocaleString()} splats`,
          );

          // Auto-center camera
          if (controlsRef.current && cameraRef.current && frameGroupRef.current) {
            const { centroid, radius } = extracted;
            const transformed = centroid
              .clone()
              .applyQuaternion(frameGroupRef.current.quaternion);
            const cam = cameraRef.current;
            const ctrl = controlsRef.current;
            const dist = Math.max(radius * 1.5, 2);
            ctrl.target.copy(transformed);
            cam.position.set(
              transformed.x + dist * 0.5,
              transformed.y + dist * 0.3,
              transformed.z - dist * 0.8,
            );
            ctrl.update();
          }
        }

        // Dispose again in case a concurrent load snuck in
        if (splatMeshRef.current) {
          frame.remove(splatMeshRef.current);
          splatMeshRef.current.dispose();
          splatMeshRef.current = null;
        }

        // Let Spark handle PLY parsing natively via fileBytes
        const mesh = new SplatMesh({ fileBytes: patched });
        frame.add(mesh);
        splatMeshRef.current = mesh;
        return mesh.initialized.then(() => {
          if (loadIdRef.current !== thisLoadId) return;
          setLoading(false);
          setNumSplats(mesh.numSplats);
        });
      })
      .catch((e: unknown) => {
        if (loadIdRef.current !== thisLoadId) return;
        console.error("Failed to load splat:", e);
        setError(String(e));
        setLoading(false);
      });
  }, []);

  /**
   * Load a scene file (SPZ preferred, PLY fallback) + optional positions sidecar.
   * If sceneUrl 404s, falls back to plyFallbackUrl (if provided).
   */
  const loadScene = useCallback(
    (sceneUrl: string, positionsUrl: string, plyFallbackUrl?: string) => {
      const frame = frameGroupRef.current;
      if (!frame) return;

      const thisLoadId = ++loadIdRef.current;

      // Dispose old splat mesh immediately
      if (splatMeshRef.current) {
        frame.remove(splatMeshRef.current);
        splatMeshRef.current.dispose();
        splatMeshRef.current = null;
      }
      positionsRef.current = null;
      kdTreeRef.current = null;

      setLoading(true);
      setError(null);
      setNumSplats(0);
      setSceneFormat(null);

      // Fetch scene file and positions sidecar in parallel
      const fetchScene = async (): Promise<{ buf: ArrayBuffer; format: string }> => {
        console.log(`[loadScene] Trying: ${sceneUrl}`);
        const res = await fetch(sceneUrl);
        if (!res.ok) {
          if (res.status === 404 && plyFallbackUrl) {
            console.log(`[loadScene] SPZ 404, falling back to PLY: ${plyFallbackUrl}`);
            const r = await fetch(plyFallbackUrl);
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return { buf: await r.arrayBuffer(), format: "ply" };
          }
          throw new Error(`HTTP ${res.status}`);
        }
        const buf = await res.arrayBuffer();
        console.log(`[loadScene] Loaded ${(buf.byteLength / 1048576).toFixed(1)}MB, magic: 0x${new DataView(buf).getUint32(0, true).toString(16)}`);
        return { buf, format: "auto" };
      };

      const posPromise = fetch(positionsUrl)
        .then((res) => (res.ok ? res.arrayBuffer() : null))
        .catch(() => null);

      Promise.all([fetchScene(), posPromise])
        .then(([sceneResult, posBuf]) => {
          if (loadIdRef.current !== thisLoadId) return;

          const { buf, format } = sceneResult;

          // Determine if this is SPZ or PLY.
          // SPZ is a gzip stream (magic 0x1f 0x8b). PLY starts with ASCII "ply\n".
          const isSPZ =
            format === "auto" &&
            buf.byteLength >= 2 &&
            new Uint8Array(buf)[0] === 0x1f &&
            new Uint8Array(buf)[1] === 0x8b;

          console.log(`[loadScene] Format: ${isSPZ ? "SPZ" : "PLY"}, size: ${(buf.byteLength / 1048576).toFixed(1)}MB, positions sidecar: ${posBuf ? (posBuf.byteLength / 1048576).toFixed(1) + "MB" : "none"}`);

          setSceneFormat({
            format: isSPZ ? "spz" : "ply",
            sizeMB: Math.round(buf.byteLength / 1048576),
          });

          // For PLY, apply header patching; SPZ passes through directly
          const fileBytes = isSPZ ? buf : patchPlyHeader(buf);

          // Parse positions: prefer sidecar, fall back to PLY extraction
          let posData: ReturnType<typeof parsePositionsSidecar> = null;
          if (posBuf) {
            posData = parsePositionsSidecar(posBuf);
          }
          if (!posData && !isSPZ) {
            posData = extractPlyPositions(fileBytes);
          }

          if (posData) {
            positionsRef.current = posData.positions;

            const t0 = performance.now();
            kdTreeRef.current = buildKDTree(posData.positions);
            console.log(
              `[useSplatScene] K-D tree built in ${(performance.now() - t0).toFixed(0)}ms for ${posData.nVerts.toLocaleString()} splats` +
                (posBuf ? " (from sidecar)" : " (from PLY)"),
            );

            // Auto-center camera
            if (controlsRef.current && cameraRef.current && frameGroupRef.current) {
              const { centroid, radius } = posData;
              const transformed = centroid
                .clone()
                .applyQuaternion(frameGroupRef.current.quaternion);
              const cam = cameraRef.current;
              const ctrl = controlsRef.current;
              const dist = Math.max(radius * 1.5, 2);
              ctrl.target.copy(transformed);
              cam.position.set(
                transformed.x + dist * 0.5,
                transformed.y + dist * 0.3,
                transformed.z - dist * 0.8,
              );
              ctrl.update();
            }
          }

          // Dispose in case a concurrent load snuck in
          if (splatMeshRef.current) {
            frame.remove(splatMeshRef.current);
            splatMeshRef.current.dispose();
            splatMeshRef.current = null;
          }

          const mesh = new SplatMesh({ fileBytes });
          frame.add(mesh);
          splatMeshRef.current = mesh;
          return mesh.initialized.then(() => {
            if (loadIdRef.current !== thisLoadId) return;
            setLoading(false);
            setNumSplats(mesh.numSplats);
          });
        })
        .catch((e: unknown) => {
          if (loadIdRef.current !== thisLoadId) return;
          console.error("Failed to load splat:", e);
          setError(String(e));
          setLoading(false);
        });
    },
    [],
  );

  const updateGizmo = useCallback(
    (centroid: THREE.Vector3 | null, mode: ToolMode) => {
      const scene = sceneRef.current;
      const camera = cameraRef.current;
      const renderer = rendererRef.current;
      const controls = controlsRef.current;

      if (!scene || !camera || !renderer || !controls) return;

      // Remove existing gizmo
      if (transformControlsRef.current) {
        transformControlsRef.current.detach();
        scene.remove(transformControlsRef.current as unknown as THREE.Object3D);
        transformControlsRef.current.dispose();
        transformControlsRef.current = null;
      }
      if (proxyMeshRef.current) {
        scene.remove(proxyMeshRef.current);
        proxyMeshRef.current.geometry.dispose();
        (proxyMeshRef.current.material as THREE.Material).dispose();
        proxyMeshRef.current = null;
      }

      if (!centroid || mode === "select") return;

      // Create invisible proxy mesh at segment centroid
      // Transform centroid through frame group (Y-down -> Y-up: negate Y and Z)
      const frameGroup = frameGroupRef.current;
      const worldCentroid = centroid.clone();
      if (frameGroup) {
        worldCentroid.applyQuaternion(frameGroup.quaternion);
      }

      const proxy = new THREE.Mesh(
        new THREE.SphereGeometry(0.02, 8, 8),
        new THREE.MeshBasicMaterial({ visible: false })
      );
      proxy.position.copy(worldCentroid);
      scene.add(proxy);
      proxyMeshRef.current = proxy;

      // Save starting transform
      proxyStartPosRef.current = proxy.position.clone();
      proxyStartQuatRef.current = proxy.quaternion.clone();
      proxyStartScaleRef.current = proxy.scale.clone();

      // Create TransformControls
      const tc = new TransformControls(camera, renderer.domElement);
      const modeMap: Record<string, "translate" | "rotate" | "scale"> = {
        translate: "translate",
        rotate: "rotate",
        scale: "scale",
      };
      tc.setMode(modeMap[mode] || "translate");
      tc.setSize(0.8);
      tc.attach(proxy);
      scene.add(tc as unknown as THREE.Object3D);
      transformControlsRef.current = tc;

      // Disable orbit controls during gizmo drag
      tc.addEventListener("dragging-changed", (event: { value: unknown }) => {
        const dragging = !!event.value;
        controls.enabled = !dragging;

        // On drag end: compute delta and fire callback
        if (!dragging) {
          const cb = onGizmoDragEndRef.current;
          if (!cb) return;

          const startPos = proxyStartPosRef.current;
          const startQuat = proxyStartQuatRef.current;
          const startScale = proxyStartScaleRef.current;
          if (!startPos || !startQuat || !startScale) return;

          const delta: GizmoDelta = {};

          if (mode === "translate") {
            const dp = new THREE.Vector3().subVectors(proxy.position, startPos);
            // Transform back through frame group inverse
            if (frameGroup) {
              const invQ = frameGroup.quaternion.clone().invert();
              dp.applyQuaternion(invQ);
            }
            delta.translation = [dp.x, dp.y, dp.z];
          } else if (mode === "rotate") {
            const dq = startQuat.clone().invert().multiply(proxy.quaternion);
            const euler = new THREE.Euler().setFromQuaternion(dq, "XYZ");
            let rx = THREE.MathUtils.radToDeg(euler.x);
            let ry = THREE.MathUtils.radToDeg(euler.y);
            let rz = THREE.MathUtils.radToDeg(euler.z);
            // Transform rotation through frame group coordinate space
            if (frameGroup) {
              ry = -ry;
              rz = -rz;
            }
            delta.rotation = [rx, ry, rz];
          } else if (mode === "scale") {
            const sx = proxy.scale.x / startScale.x;
            const sy = proxy.scale.y / startScale.y;
            const sz = proxy.scale.z / startScale.z;
            delta.scale = [sx, sy, sz];
          }

          cb(delta);

          // Reset proxy for next drag
          proxyStartPosRef.current = proxy.position.clone();
          proxyStartQuatRef.current = proxy.quaternion.clone();
          proxyStartScaleRef.current = proxy.scale.clone();
        }
      });
    },
    []
  );

  return {
    containerRef,
    loading,
    error,
    numSplats,
    sceneFormat,
    loadPly,
    loadScene,
    splatMeshRef,
    positionsRef,
    sceneRef,
    cameraRef,
    rendererRef,
    onPickRef,
    onHoverRef,
    onGizmoDragEndRef,
    updateGizmo,
  };
}
