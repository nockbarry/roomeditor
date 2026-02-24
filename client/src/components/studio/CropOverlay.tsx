import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useEditorStore } from "../../stores/editorStore.ts";
import type { CropBox, CropSphere } from "../../hooks/useCropTool.ts";

interface CropOverlayProps {
  scene: THREE.Scene | null;
  cropBox: CropBox | null;
  cropSphere: CropSphere | null;
  onBoxChange?: (box: CropBox) => void;
  onSphereChange?: (sphere: CropSphere) => void;
}

/**
 * Renders a translucent wireframe box or sphere in the 3D scene
 * to preview the crop volume before applying.
 */
export function CropOverlay({
  scene,
  cropBox,
  cropSphere,
}: CropOverlayProps) {
  const toolMode = useEditorStore((s) => s.toolMode);
  const cropMode = useEditorStore((s) => s.cropMode);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const wireRef = useRef<THREE.LineSegments | null>(null);

  // Color: red for delete-inside, blue for delete-outside
  const color = cropMode === "delete-inside" ? 0xff4444 : 0x4488ff;

  useEffect(() => {
    if (!scene) return;

    // Clean up previous
    if (meshRef.current) {
      scene.remove(meshRef.current);
      meshRef.current.geometry.dispose();
      (meshRef.current.material as THREE.Material).dispose();
      meshRef.current = null;
    }
    if (wireRef.current) {
      scene.remove(wireRef.current);
      wireRef.current.geometry.dispose();
      (wireRef.current.material as THREE.Material).dispose();
      wireRef.current = null;
    }

    if (toolMode === "crop-box" && cropBox) {
      const size = [
        cropBox.max[0] - cropBox.min[0],
        cropBox.max[1] - cropBox.min[1],
        cropBox.max[2] - cropBox.min[2],
      ] as const;
      const center = [
        (cropBox.min[0] + cropBox.max[0]) / 2,
        (cropBox.min[1] + cropBox.max[1]) / 2,
        (cropBox.min[2] + cropBox.max[2]) / 2,
      ] as const;

      const geo = new THREE.BoxGeometry(size[0], size[1], size[2]);

      // Translucent fill
      const mat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(center[0], center[1], center[2]);
      mesh.renderOrder = 998;
      scene.add(mesh);
      meshRef.current = mesh;

      // Wireframe edges
      const edges = new THREE.EdgesGeometry(geo);
      const lineMat = new THREE.LineBasicMaterial({ color, linewidth: 1, transparent: true, opacity: 0.6 });
      const wire = new THREE.LineSegments(edges, lineMat);
      wire.position.set(center[0], center[1], center[2]);
      wire.renderOrder = 999;
      scene.add(wire);
      wireRef.current = wire;
    } else if (toolMode === "crop-sphere" && cropSphere) {
      const geo = new THREE.SphereGeometry(cropSphere.radius, 24, 16);

      const mat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(cropSphere.center[0], cropSphere.center[1], cropSphere.center[2]);
      mesh.renderOrder = 998;
      scene.add(mesh);
      meshRef.current = mesh;

      // Wireframe
      const edges = new THREE.EdgesGeometry(geo);
      const lineMat = new THREE.LineBasicMaterial({ color, linewidth: 1, transparent: true, opacity: 0.6 });
      const wire = new THREE.LineSegments(edges, lineMat);
      wire.position.set(cropSphere.center[0], cropSphere.center[1], cropSphere.center[2]);
      wire.renderOrder = 999;
      scene.add(wire);
      wireRef.current = wire;
    }

    return () => {
      if (meshRef.current && scene) {
        scene.remove(meshRef.current);
        meshRef.current.geometry.dispose();
        (meshRef.current.material as THREE.Material).dispose();
        meshRef.current = null;
      }
      if (wireRef.current && scene) {
        scene.remove(wireRef.current);
        wireRef.current.geometry.dispose();
        (wireRef.current.material as THREE.Material).dispose();
        wireRef.current = null;
      }
    };
  }, [scene, toolMode, cropBox, cropSphere, color]);

  return null;
}
