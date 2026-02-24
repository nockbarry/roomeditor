import { describe, it, expect } from "vitest";
import { buildKDTree, queryNearest, queryRadius, queryBox, type KDTreeNode } from "./useKDTree";

/** Brute-force nearest neighbor for reference validation. */
function bruteForceNearest(positions: Float32Array, px: number, py: number, pz: number): number {
  let bestIdx = -1;
  let bestDistSq = Infinity;
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    const dx = positions[i * 3] - px;
    const dy = positions[i * 3 + 1] - py;
    const dz = positions[i * 3 + 2] - pz;
    const distSq = dx * dx + dy * dy + dz * dz;
    if (distSq < bestDistSq) {
      bestDistSq = distSq;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function distSq(positions: Float32Array, idx: number, px: number, py: number, pz: number): number {
  const dx = positions[idx * 3] - px;
  const dy = positions[idx * 3 + 1] - py;
  const dz = positions[idx * 3 + 2] - pz;
  return dx * dx + dy * dy + dz * dz;
}

describe("buildKDTree", () => {
  it("returns null for empty input", () => {
    const tree = buildKDTree(new Float32Array(0));
    expect(tree).toBeNull();
  });

  it("builds a leaf node for a single point", () => {
    const positions = new Float32Array([1, 2, 3]);
    const tree = buildKDTree(positions);
    expect(tree).not.toBeNull();
    expect(tree!.idx).toBe(0);
    expect(tree!.left).toBeNull();
    expect(tree!.right).toBeNull();
  });

  it("builds a valid tree for two points", () => {
    const positions = new Float32Array([0, 0, 0, 1, 1, 1]);
    const tree = buildKDTree(positions);
    expect(tree).not.toBeNull();
    // Should have one child (the other point)
    const childCount = (tree!.left ? 1 : 0) + (tree!.right ? 1 : 0);
    expect(childCount).toBe(1);
  });
});

describe("queryNearest", () => {
  it("returns -1 for null tree", () => {
    const result = queryNearest(null, new Float32Array(0), 0, 0, 0);
    expect(result).toBe(-1);
  });

  it("always returns 0 for single-point tree", () => {
    const positions = new Float32Array([5, 10, 15]);
    const tree = buildKDTree(positions)!;
    expect(queryNearest(tree, positions, 0, 0, 0)).toBe(0);
    expect(queryNearest(tree, positions, 100, 100, 100)).toBe(0);
  });

  it("finds exact matches in a small set", () => {
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      1, 1, 1,
    ]);
    const tree = buildKDTree(positions)!;

    for (let i = 0; i < 5; i++) {
      const result = queryNearest(tree, positions, positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      expect(result).toBe(i);
    }
  });

  it("finds approximate nearest neighbor", () => {
    const positions = new Float32Array([
      0, 0, 0,
      10, 0, 0,
      0, 10, 0,
      0, 0, 10,
    ]);
    const tree = buildKDTree(positions)!;

    // Query near (10,0,0) should return index 1
    expect(queryNearest(tree, positions, 9.5, 0.1, -0.1)).toBe(1);
    // Query near (0,10,0) should return index 2
    expect(queryNearest(tree, positions, 0.2, 9.8, 0.1)).toBe(2);
    // Query near origin should return index 0
    expect(queryNearest(tree, positions, 0.1, 0.1, 0.1)).toBe(0);
  });

  it("handles duplicate points without crashing", () => {
    const positions = new Float32Array([
      1, 2, 3,
      1, 2, 3,
      1, 2, 3,
      5, 5, 5,
    ]);
    const tree = buildKDTree(positions)!;

    // Query at the duplicate location — should return one of 0, 1, 2
    const result = queryNearest(tree, positions, 1, 2, 3);
    expect(result).toBeGreaterThanOrEqual(0);
    expect(result).toBeLessThanOrEqual(2);

    // Query near (5,5,5) should return index 3
    expect(queryNearest(tree, positions, 5, 5, 5)).toBe(3);
  });

  it("handles collinear points correctly", () => {
    // All points along the x-axis — degenerate geometry
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      2, 0, 0,
      3, 0, 0,
      4, 0, 0,
    ]);
    const tree = buildKDTree(positions)!;

    expect(queryNearest(tree, positions, 0.4, 0, 0)).toBe(0);
    expect(queryNearest(tree, positions, 2.6, 0, 0)).toBe(3);
    expect(queryNearest(tree, positions, 3.9, 0, 0)).toBe(4);
  });

  it("matches brute-force on random data (500 points, 50 queries)", () => {
    // Seeded pseudo-random for determinism
    let seed = 42;
    function rand(): number {
      seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
      return seed / 0x7fffffff;
    }

    const n = 500;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) {
      positions[i] = (rand() - 0.5) * 100;
    }

    const tree = buildKDTree(positions)!;

    for (let q = 0; q < 50; q++) {
      const px = (rand() - 0.5) * 120;
      const py = (rand() - 0.5) * 120;
      const pz = (rand() - 0.5) * 120;

      const kdResult = queryNearest(tree, positions, px, py, pz);
      const bfResult = bruteForceNearest(positions, px, py, pz);

      // Distances should match (indices may differ if equidistant)
      const kdDist = distSq(positions, kdResult, px, py, pz);
      const bfDist = distSq(positions, bfResult, px, py, pz);
      expect(kdDist).toBeCloseTo(bfDist, 5);
    }
  });
});

// ---------------------------------------------------------------------------
// queryRadius
// ---------------------------------------------------------------------------

/** Brute-force radius search for reference validation. */
function bruteForceRadius(
  positions: Float32Array, cx: number, cy: number, cz: number, radius: number,
): Set<number> {
  const result = new Set<number>();
  const n = positions.length / 3;
  const r2 = radius * radius;
  for (let i = 0; i < n; i++) {
    const dx = positions[i * 3] - cx;
    const dy = positions[i * 3 + 1] - cy;
    const dz = positions[i * 3 + 2] - cz;
    if (dx * dx + dy * dy + dz * dz <= r2) result.add(i);
  }
  return result;
}

describe("queryRadius", () => {
  it("returns empty for null tree", () => {
    const result = queryRadius(null, new Float32Array(0), 0, 0, 0, 1);
    expect(result.length).toBe(0);
  });

  it("returns single point when within radius", () => {
    const positions = new Float32Array([1, 0, 0]);
    const tree = buildKDTree(positions)!;
    expect(queryRadius(tree, positions, 0, 0, 0, 2).length).toBe(1);
    expect(queryRadius(tree, positions, 0, 0, 0, 0.5).length).toBe(0);
  });

  it("finds all points within radius", () => {
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      2, 0, 0,
      3, 0, 0,
      10, 0, 0,
    ]);
    const tree = buildKDTree(positions)!;
    const result = queryRadius(tree, positions, 1.5, 0, 0, 1.6);
    const indices = new Set(Array.from(result));
    // Should find 0, 1, 2, 3 (distances: 1.5, 0.5, 0.5, 1.5) — all ≤ 1.6
    expect(indices.has(0)).toBe(true);
    expect(indices.has(1)).toBe(true);
    expect(indices.has(2)).toBe(true);
    expect(indices.has(3)).toBe(true);
    expect(indices.has(4)).toBe(false); // distance = 8.5
  });

  it("matches brute-force on random data (300 points, 20 queries)", () => {
    let seed = 123;
    function rand(): number {
      seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
      return seed / 0x7fffffff;
    }

    const n = 300;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) {
      positions[i] = (rand() - 0.5) * 50;
    }
    const tree = buildKDTree(positions)!;

    for (let q = 0; q < 20; q++) {
      const cx = (rand() - 0.5) * 60;
      const cy = (rand() - 0.5) * 60;
      const cz = (rand() - 0.5) * 60;
      const radius = rand() * 15 + 1;

      const kdResult = new Set(Array.from(queryRadius(tree, positions, cx, cy, cz, radius)));
      const bfResult = bruteForceRadius(positions, cx, cy, cz, radius);

      expect(kdResult.size).toBe(bfResult.size);
      for (const idx of bfResult) {
        expect(kdResult.has(idx)).toBe(true);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// queryBox
// ---------------------------------------------------------------------------

/** Brute-force box search for reference validation. */
function bruteForceBox(
  positions: Float32Array,
  minPt: [number, number, number],
  maxPt: [number, number, number],
): Set<number> {
  const result = new Set<number>();
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (x >= minPt[0] && x <= maxPt[0] &&
        y >= minPt[1] && y <= maxPt[1] &&
        z >= minPt[2] && z <= maxPt[2]) {
      result.add(i);
    }
  }
  return result;
}

describe("queryBox", () => {
  it("returns empty for null tree", () => {
    const result = queryBox(null, new Float32Array(0), [0, 0, 0], [1, 1, 1]);
    expect(result.length).toBe(0);
  });

  it("returns single point when inside box", () => {
    const positions = new Float32Array([0.5, 0.5, 0.5]);
    const tree = buildKDTree(positions)!;
    expect(queryBox(tree, positions, [0, 0, 0], [1, 1, 1]).length).toBe(1);
    expect(queryBox(tree, positions, [1, 1, 1], [2, 2, 2]).length).toBe(0);
  });

  it("finds points inside a box", () => {
    const positions = new Float32Array([
      0, 0, 0,
      1, 1, 1,
      2, 2, 2,
      -1, -1, -1,
      5, 5, 5,
    ]);
    const tree = buildKDTree(positions)!;
    const result = queryBox(tree, positions, [-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]);
    const indices = new Set(Array.from(result));
    expect(indices.has(0)).toBe(true);
    expect(indices.has(1)).toBe(true);
    expect(indices.has(2)).toBe(false);
    expect(indices.has(3)).toBe(false);
    expect(indices.has(4)).toBe(false);
  });

  it("returns all points for large enough box", () => {
    const positions = new Float32Array([
      0, 0, 0,  1, 1, 1,  -1, -1, -1,
    ]);
    const tree = buildKDTree(positions)!;
    const result = queryBox(tree, positions, [-10, -10, -10], [10, 10, 10]);
    expect(result.length).toBe(3);
  });

  it("matches brute-force on random data (300 points, 20 queries)", () => {
    let seed = 456;
    function rand(): number {
      seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
      return seed / 0x7fffffff;
    }

    const n = 300;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) {
      positions[i] = (rand() - 0.5) * 50;
    }
    const tree = buildKDTree(positions)!;

    for (let q = 0; q < 20; q++) {
      const x1 = (rand() - 0.5) * 60;
      const y1 = (rand() - 0.5) * 60;
      const z1 = (rand() - 0.5) * 60;
      const x2 = x1 + rand() * 20;
      const y2 = y1 + rand() * 20;
      const z2 = z1 + rand() * 20;

      const minPt: [number, number, number] = [x1, y1, z1];
      const maxPt: [number, number, number] = [x2, y2, z2];

      const kdResult = new Set(Array.from(queryBox(tree, positions, minPt, maxPt)));
      const bfResult = bruteForceBox(positions, minPt, maxPt);

      expect(kdResult.size).toBe(bfResult.size);
      for (const idx of bfResult) {
        expect(kdResult.has(idx)).toBe(true);
      }
    }
  });
});
