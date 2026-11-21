"""Minimal PCD reader/writer (ascii / binary / binary_compressed) and PLY exporter.

Only numpy is required. The reader supports the PCD v0.7 formats produced by
PCL / CloudCompare / ROS pcl_ros, including LZF-compressed binary files.
"""
import struct

import numpy as np

TYPE_MAP = {
    "F": np.float32,
    "U": np.uint8,
    "I": np.int32,
    "D": np.float64,
    "S": np.uint16,
}


def _lzf_decompress(data, expected_len):
    out = bytearray(expected_len)
    ip = op = 0
    n = len(data)
    while ip < n:
        ctrl = data[ip]
        ip += 1
        if ctrl < 32:
            run = ctrl + 1
            out[op:op + run] = data[ip:ip + run]
            ip += run
            op += run
        else:
            length = ctrl >> 5
            ref = op - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                length += data[ip]
                ip += 1
            ref -= data[ip]
            ip += 1
            for _ in range(length + 2):
                out[op] = out[ref]
                op += 1
                ref += 1
    return bytes(out[:op])


def read_pcd(path):
    """Return a structured ndarray (binary) or (N, n_fields) float32 array (ascii)."""
    with open(path, "rb") as f:
        fields = sizes = types = counts = None
        npts = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("no DATA line: %s" % path)
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("DATA"):
                mode = s.split()[1].lower()
                break
            p = s.split()
            if not p:
                continue
            k = p[0].upper()
            if k == "FIELDS":
                fields = p[1:]
            elif k == "SIZE":
                sizes = [int(v) for v in p[1:]]
            elif k == "TYPE":
                types = p[1:]
            elif k == "COUNT":
                counts = [int(v) for v in p[1:]]
            elif k == "POINTS":
                npts = int(p[1:][0])
        nf = len(fields)
        counts = counts or [1] * nf
        types = types or ["F"] * nf
        sizes = sizes or [4] * nf
        dtype = np.dtype(
            [(f, TYPE_MAP[t], (c,) if c > 1 else ()) for f, t, c in zip(fields, types, counts)]
        )
        if mode == "ascii":
            arr = np.fromstring(f.read().decode("ascii", "ignore"), sep=" ")
            cols = sum(counts)
            return arr[: len(arr) // cols * cols].reshape(-1, cols).astype(np.float32)
        if mode == "binary":
            return np.frombuffer(f.read(), dtype=dtype, count=npts).copy()
        if mode == "binary_compressed":
            cs, us = struct.unpack("II", f.read(8))
            raw = _lzf_decompress(f.read(cs), us)
            return np.frombuffer(raw, dtype=dtype, count=npts).copy()
        raise ValueError("unsupported DATA mode: %s" % mode)


def as_xyz(rec):
    """Normalise read_pcd() output to an (N, 3) float64 xyz array."""
    if getattr(rec, "dtype", None) is not None and rec.dtype.names:
        return np.column_stack(
            [np.asarray(rec["x"], float), np.asarray(rec["y"], float), np.asarray(rec["z"], float)]
        )
    return np.asarray(rec[:, :3], float)


def load_pcd_xyz(path):
    return as_xyz(read_pcd(path))


def write_pcd(path, xyz, extra=None):
    """Write binary PCD v0.7. extra: dict{name: (N,) float32}."""
    xyz = np.asarray(xyz, np.float32)
    n = len(xyz)
    cols = [("x y z", xyz.astype(np.float64))]
    fields = ["x", "y", "z"]
    sizes = [4, 4, 4]
    if extra:
        for name, arr in extra.items():
            fields.append(name)
            sizes.append(4)
            cols.append([np.asarray(arr, np.float32)])
    dt = np.dtype([(f, "<f4") for f in fields])
    data = np.zeros(n, dt)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if extra:
        for name, arr in extra.items():
            data[name] = np.asarray(arr, np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS %s\n" % " ".join(fields)
        + "SIZE %s\n" % " ".join(["4"] * len(fields))
        + "TYPE %s\n" % " ".join(["F"] * len(fields))
        + "COUNT %s\n" % " ".join(["1"] * len(fields))
        + "WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n" % (n, n)
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def write_ply_colored(path, xyz, rgb):
    """Write binary PLY with per-vertex colors (rgb: (N,3) uint8), handy for Blender."""
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    n = len(xyz)
    dt = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    data = np.zeros(n, dt)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex %d\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n" % n
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def height_colormap(z, zmax=2.5):
    """Simple turbo-like colormap implemented with matplotlib, returns uint8 RGB."""
    from matplotlib import cm

    return (cm.turbo(np.clip(np.asarray(z) / zmax, 0.0, 1.0))[:, :3] * 255).astype(np.uint8)
