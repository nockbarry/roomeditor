"""Read/write Gaussian Splatting PLY files."""

import struct
from pathlib import Path

import numpy as np


def read_ply_header(path: Path) -> tuple[int, list[str], int]:
    """Read PLY header and return (vertex_count, property_names, header_bytes).

    Returns:
        vertex_count: Number of vertices
        properties: List of property names
        header_size: Size of header in bytes
    """
    properties = []
    vertex_count = 0

    with open(path, "rb") as f:
        while True:
            line = f.readline()
            line_str = line.decode("ascii", errors="ignore").strip()

            if line_str.startswith("element vertex"):
                vertex_count = int(line_str.split()[-1])
            elif line_str.startswith("property"):
                parts = line_str.split()
                if len(parts) >= 3:
                    properties.append(parts[2])
            elif line_str == "end_header":
                header_size = f.tell()
                break

    return vertex_count, properties, header_size


def read_ply_positions(path: Path) -> np.ndarray:
    """Read just the XYZ positions from a Gaussian Splatting PLY file.

    Returns an (N, 3) float32 array.
    """
    vertex_count, properties, header_size = read_ply_header(path)

    # Calculate bytes per vertex (all floats)
    bytes_per_vertex = len(properties) * 4  # assuming all float32

    # Find x, y, z indices
    x_idx = properties.index("x") if "x" in properties else 0
    y_idx = properties.index("y") if "y" in properties else 1
    z_idx = properties.index("z") if "z" in properties else 2

    positions = np.zeros((vertex_count, 3), dtype=np.float32)

    with open(path, "rb") as f:
        f.seek(header_size)
        for i in range(vertex_count):
            vertex_data = struct.unpack(f"<{len(properties)}f", f.read(bytes_per_vertex))
            positions[i] = [vertex_data[x_idx], vertex_data[y_idx], vertex_data[z_idx]]

    return positions
