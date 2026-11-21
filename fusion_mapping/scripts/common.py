"""Geometry helpers shared by the fusion pipeline.

All functions are pure numpy/scipy: no ROS, no Open3D.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------
# downsampling / filtering
# ----------------------------------------------------------------------------
def voxel_downsample(pts, voxel):
    """Keep one representative point per voxel (deterministic: first index)."""
    key = np.floor(np.asarray(pts) / float(voxel)).astype(np.int64)
    key -= key.min(axis=0, keepdims=True)
    dims = key.max(axis=0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    _, idx = np.unique(lin, return_index=True)
    return pts[idx]


def connected_component_filter(pts, voxel=0.1, min_points=100):
    """Remove isolated blobs: voxelise, 26-connected components, drop small ones.

    This is the main de-fogging step. Fog / moving people / robot ghost trails
    form small components; real structures are large and connected.
    """
    key = np.floor(np.asarray(pts) / float(voxel)).astype(np.int64)
    origin = key.min(axis=0) - 1
    shape = tuple((key.max(axis=0) - origin + 1).tolist())
    occ = np.zeros(shape, bool)
    occ[tuple((key - origin).T.tolist())] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if n == 0:
        return pts
    sizes = ndimage.sum(occ, lab, range(1, n + 1))
    keep_ids = np.nonzero(sizes >= min_points)[0] + 1
    keep = np.isin(lab, keep_ids)
    return pts[keep[tuple((key - origin).T.tolist())]]


def fit_plane(pts):
    """Total-least-squares plane fit -> (centroid, unit normal)."""
    c = pts.mean(axis=0)
    _, v = np.linalg.eigh((pts - c).T @ (pts - c))
    return c, v[:, 0]


def flatten_ground(pts, band=0.35, robust=True):
    """Fit a plane to near-ground points (|z| < band) and subtract it.

    Returns (flattened_points, plane_coeffs[row a, b, c of z = a*x + b*y + c]).
    """
    m = np.abs(pts[:, 2]) < band
    if m.sum() < 100:
        return pts, np.zeros(3)
    A = np.c_[pts[m, 0], pts[m, 1], np.ones(m.sum())]
    if robust:
        coef, *_ = np.linalg.lstsq(A, pts[m, 2], rcond=None)
        for _ in range(3):
            res = pts[m, 2] - A @ coef
            keep = np.abs(res) < max(0.05, np.percentile(np.abs(res), 80))
            coef, *_ = np.linalg.lstsq(A[keep], pts[m, 2][keep], rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(A, pts[m, 2], rcond=None)
    out = pts.copy()
    out[:, 2] -= coef[0] * out[:, 0] + coef[1] * out[:, 1] + coef[2]
    return out, coef


# ----------------------------------------------------------------------------
# SE(3) estimation
# ----------------------------------------------------------------------------
def umeyama(src, dst, with_scale=False):
    """Least-squares similarity transform: dst ~= s * R @ src + t."""
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    C = (dst - cd).T @ (src - cs) / max(len(src), 1)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    if with_scale:
        var = ((src - cs) ** 2).sum()
        s = float(S.sum() / var) if var > 1e-12 else 1.0
    else:
        s = 1.0
    t = cd - s * R @ cs
    return s, R, t


def se3_exp(xi):
    """Exponential map of a 6-vector twist [v(3), w(3)] -> 4x4 matrix."""
    v, w = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        R = np.eye(3)
        V = np.eye(3)
    else:
        k = w / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        V = np.eye(3) + (1 - np.cos(th)) / th * K + (1 - np.sin(th) / th) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def transform_points(T, pts):
    pts = np.asarray(pts, float)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def estimate_normals(pts, k=16):
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)), workers=-1)
    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        nb = pts[idx[i]]
        nb = nb - nb.mean(axis=0)
        _, v = np.linalg.eigh(nb.T @ nb)
        normals[i] = v[:, 0]
    return normals


# ----------------------------------------------------------------------------
# ICP
# ----------------------------------------------------------------------------
def icp_point_to_point(src, tgt, T0=None, max_iter=40, trim=0.7, verbose=False):
    """Trimmed point-to-point ICP (rigid). Returns (T, median_residual)."""
    T = np.eye(4) if T0 is None else np.asarray(T0, float).copy()
    tree = cKDTree(tgt)
    for it in range(max_iter):
        q = transform_points(T, src)
        d, idx = tree.query(q, k=1, workers=-1)
        m = d < np.quantile(d, trim)
        if m.sum() < 100:
            break
        _, dR, dt = umeyama(q[m], tgt[idx[m]])
        dT = np.eye(4)
        dT[:3, :3], dT[:3, 3] = dR, dt
        T = dT @ T
        if verbose and it % 10 == 0:
            print("  p2p icp it%02d: median %.4f m" % (it, np.median(d[m])))
    q = transform_points(T, src)
    d, _ = tree.query(q, k=1, workers=-1)
    return T, float(np.median(d))


def icp_point_to_plane(src, tgt, tgt_normals, T0=None, max_iter=60, trim=0.7):
    """Trimmed point-to-plane ICP with correct Jacobian and line search.

    Linearised: residual r_i = n_i . (T q_i - y_i), Jacobian row = [n, q x n]
    for the twist (v, w) applied on the left (world frame). A step is accepted
    only when the median NN distance improves, which keeps the solver stable on
    partially overlapping venue scans.
    """
    T = np.eye(4) if T0 is None else np.asarray(T0, float).copy()
    tree = cKDTree(tgt)

    def median_dist(Tcur):
        q = transform_points(Tcur, src)
        d, _ = tree.query(q, k=1, workers=-1)
        return float(np.median(d))

    for it in range(max_iter):
        q = transform_points(T, src)
        d, idx = tree.query(q, k=1, workers=-1)
        m = d < np.quantile(d, trim)
        n = tgt_normals[idx[m]]
        r = ((q[m] - tgt[idx[m]]) * n).sum(axis=1)
        A = np.hstack([n, np.cross(q[m], n)])
        x, *_ = np.linalg.lstsq(A, -r, rcond=None)
        T2 = se3_exp(x) @ T
        if median_dist(T2) < median_dist(T):
            T = T2
        else:
            break
    return T, median_dist(T), it


# ----------------------------------------------------------------------------
# 2D occupancy utilities (fast rotation/translation search)
# ----------------------------------------------------------------------------
def occupancy_2d(pts, res, zmin=None, zmax=None, pad_cells=None):
    """Rasterise XY occupancy at `res` metres. Returns (grid, origin_xy)."""
    pts = np.asarray(pts, float)
    if zmin is not None or zmax is not None:
        m = np.ones(len(pts), bool)
        if zmin is not None:
            m &= pts[:, 2] > zmin
        if zmax is not None:
            m &= pts[:, 2] < zmax
        pts = pts[m]
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    nx = int((pts[:, 0].max() - x0) / res) + 2
    ny = int((pts[:, 1].max() - y0) / res) + 2
    g = np.zeros((ny, nx), np.float32)
    i = np.clip(((pts[:, 1] - y0) / res).astype(int), 0, ny - 1)
    j = np.clip(((pts[:, 0] - x0) / res).astype(int), 0, nx - 1)
    np.add.at(g, (i, j), 1)
    g = ndimage.gaussian_filter((g > 0).astype(np.float32), 1.0)
    return g, np.array([x0, y0])
