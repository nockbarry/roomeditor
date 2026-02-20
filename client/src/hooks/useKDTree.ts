/**
 * K-D tree for fast nearest-neighbor lookups on a Float32Array of positions.
 * Operates directly on the positions buffer via index arrays (no copying).
 */

export interface KDTreeNode {
  /** Splat index (leaf) or -1 (internal node) */
  idx: number;
  /** Split axis: 0=x, 1=y, 2=z */
  axis: number;
  /** Split value */
  split: number;
  left: KDTreeNode | null;
  right: KDTreeNode | null;
}

/**
 * Build a k-d tree from interleaved xyz positions (Float32Array of length n*3).
 * Returns the root node. Build time ~1-2s for 4.6M points.
 */
export function buildKDTree(positions: Float32Array): KDTreeNode | null {
  const n = positions.length / 3;
  if (n === 0) return null;

  // Create index array
  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;

  return buildNode(positions, indices, 0, n, 0);
}

function buildNode(
  positions: Float32Array,
  indices: Uint32Array,
  start: number,
  end: number,
  depth: number,
): KDTreeNode {
  const count = end - start;

  if (count === 1) {
    return { idx: indices[start], axis: 0, split: 0, left: null, right: null };
  }

  const axis = depth % 3;

  // Find median using nth_element-style partial sort
  const mid = start + (count >> 1);
  nthElement(positions, indices, start, end, mid, axis);

  const splitIdx = indices[mid];
  const split = positions[splitIdx * 3 + axis];

  const node: KDTreeNode = {
    idx: splitIdx,
    axis,
    split,
    left: null,
    right: null,
  };

  if (mid > start) {
    node.left = buildNode(positions, indices, start, mid, depth + 1);
  }
  if (mid + 1 < end) {
    node.right = buildNode(positions, indices, mid + 1, end, depth + 1);
  }

  return node;
}

/** Partial sort so that indices[k] is the median along the given axis. */
function nthElement(
  positions: Float32Array,
  indices: Uint32Array,
  left: number,
  right: number,
  k: number,
  axis: number,
): void {
  while (right - left > 1) {
    // Pick pivot as median of three
    const mid = left + ((right - left) >> 1);
    const a = positions[indices[left] * 3 + axis];
    const b = positions[indices[mid] * 3 + axis];
    const c = positions[indices[right - 1] * 3 + axis];

    let pivotPos: number;
    if ((a <= b && b <= c) || (c <= b && b <= a)) pivotPos = mid;
    else if ((b <= a && a <= c) || (c <= a && a <= b)) pivotPos = left;
    else pivotPos = right - 1;

    // Move pivot to end
    const tmp = indices[pivotPos];
    indices[pivotPos] = indices[right - 1];
    indices[right - 1] = tmp;
    const pivotVal = positions[tmp * 3 + axis];

    let store = left;
    for (let i = left; i < right - 1; i++) {
      if (positions[indices[i] * 3 + axis] < pivotVal) {
        const t = indices[i];
        indices[i] = indices[store];
        indices[store] = t;
        store++;
      }
    }
    // Move pivot to its final position
    const t2 = indices[store];
    indices[store] = indices[right - 1];
    indices[right - 1] = t2;

    if (store === k) return;
    if (k < store) right = store;
    else left = store + 1;
  }
}

/**
 * Find the nearest splat index to the query point (px, py, pz).
 * Returns the splat index, or -1 if tree is null.
 */
export function queryNearest(
  tree: KDTreeNode | null,
  positions: Float32Array,
  px: number,
  py: number,
  pz: number,
): number {
  if (!tree) return -1;

  let bestIdx = -1;
  let bestDistSq = Infinity;
  const query = [px, py, pz];

  function search(node: KDTreeNode | null): void {
    if (!node) return;

    // Distance to this node's point
    const ni = node.idx;
    const dx = positions[ni * 3] - px;
    const dy = positions[ni * 3 + 1] - py;
    const dz = positions[ni * 3 + 2] - pz;
    const distSq = dx * dx + dy * dy + dz * dz;

    if (distSq < bestDistSq) {
      bestDistSq = distSq;
      bestIdx = ni;
    }

    // Which side of the split plane is the query on?
    const diff = query[node.axis] - node.split;
    const first = diff < 0 ? node.left : node.right;
    const second = diff < 0 ? node.right : node.left;

    // Search the nearer side first
    search(first);

    // Only search the other side if the splitting plane is closer than current best
    if (diff * diff < bestDistSq) {
      search(second);
    }
  }

  search(tree);
  return bestIdx;
}
