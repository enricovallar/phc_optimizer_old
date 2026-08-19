"""
Parametric Locus & Manifold Extraction for Optimization Landscapes
Uses standard corroborated scientific tools (scipy, numpy) for:
- Percentile thresholding & morphological closing (scipy.ndimage)
- Connected component labeling & segmentation (scipy.ndimage.label)
- Topological skeletonization (Zhang-Suen / skimage.morphology)
- Endpoint-to-endpoint graph path ordering
- Parametric B-spline curve fitting & smoothing (scipy.interpolate.splprep/splev)
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import splprep, splev


def zhang_suen_thinning(image: np.ndarray) -> np.ndarray:
    """
    Pure NumPy implementation of the standard Zhang-Suen morphological thinning
    algorithm for topological skeletonization. Preserves connectivity and endpoints.
    """
    img = image.copy().astype(np.uint8)
    prev = np.zeros_like(img)

    while True:
        # Step 1
        p2 = np.roll(img, -1, axis=0)
        p3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
        p4 = np.roll(img, 1, axis=1)
        p5 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
        p6 = np.roll(img, 1, axis=0)
        p7 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
        p8 = np.roll(img, -1, axis=1)
        p9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(int)
            + ((p3 == 0) & (p4 == 1)).astype(int)
            + ((p4 == 0) & (p5 == 1)).astype(int)
            + ((p5 == 0) & (p6 == 1)).astype(int)
            + ((p6 == 0) & (p7 == 1)).astype(int)
            + ((p7 == 0) & (p8 == 1)).astype(int)
            + ((p8 == 0) & (p9 == 1)).astype(int)
            + ((p9 == 0) & (p2 == 1)).astype(int)
        )

        m1 = (
            (img == 1)
            & (neighbors >= 2)
            & (neighbors <= 6)
            & (transitions == 1)
            & (p2 * p4 * p6 == 0)
            & (p4 * p6 * p8 == 0)
        )
        img[m1] = 0

        # Step 2
        p2 = np.roll(img, -1, axis=0)
        p3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
        p4 = np.roll(img, 1, axis=1)
        p5 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
        p6 = np.roll(img, 1, axis=0)
        p7 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
        p8 = np.roll(img, -1, axis=1)
        p9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(int)
            + ((p3 == 0) & (p4 == 1)).astype(int)
            + ((p4 == 0) & (p5 == 1)).astype(int)
            + ((p5 == 0) & (p6 == 1)).astype(int)
            + ((p6 == 0) & (p7 == 1)).astype(int)
            + ((p7 == 0) & (p8 == 1)).astype(int)
            + ((p8 == 0) & (p9 == 1)).astype(int)
            + ((p9 == 0) & (p2 == 1)).astype(int)
        )

        m2 = (
            (img == 1)
            & (neighbors >= 2)
            & (neighbors <= 6)
            & (transitions == 1)
            & (p2 * p4 * p8 == 0)
            & (p2 * p6 * p8 == 0)
        )
        img[m2] = 0

        if np.array_equal(img, prev):
            break
        prev = img.copy()

    return img.astype(bool)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Thin a binary 2D mask to a 1-pixel-wide skeleton.
    Uses skimage.morphology.skeletonize if available, otherwise built-in Zhang-Suen.
    """
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask)
    except Exception:
        return zhang_suen_thinning(mask)


def order_skeleton_points(pts_x: np.ndarray, pts_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orders 2D skeleton pixel coordinates into a continuous sequential trajectory,
    handling both open branches and complete 360-degree closed circular loops.
    """
    if len(pts_x) <= 2:
        return pts_x, pts_y

    n = len(pts_x)

    # Estimate local 8-connectivity neighbor distance threshold
    diffs_x = np.abs(np.subtract.outer(pts_x, pts_x))
    diffs_y = np.abs(np.subtract.outer(pts_y, pts_y))
    nonzero_x = diffs_x[diffs_x > 1e-9]
    nonzero_y = diffs_y[diffs_y > 1e-9]
    dx = float(np.min(nonzero_x)) if nonzero_x.size > 0 else 1.0
    dy = float(np.min(nonzero_y)) if nonzero_y.size > 0 else 1.0
    diag_step = 1.5 * np.hypot(dx, dy)

    # Build adjacency list
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pts_x[i] - pts_x[j], pts_y[i] - pts_y[j])
            if d <= diag_step:
                adj[i].append(j)
                adj[j].append(i)

    # Identify degree-1 endpoints (if open branch)
    deg1 = [i for i, nbrs in enumerate(adj) if len(nbrs) == 1]
    start = deg1[0] if deg1 else 0

    visited = set([start])
    path = [start]
    curr = start
    while len(visited) < n:
        unvisited_nbrs = [nbr for nbr in adj[curr] if nbr not in visited]
        if unvisited_nbrs:
            next_node = min(unvisited_nbrs, key=lambda j: np.hypot(pts_x[curr] - pts_x[j], pts_y[curr] - pts_y[j]))
        else:
            unvisited_all = [i for i in range(n) if i not in visited]
            next_node = min(unvisited_all, key=lambda j: np.hypot(pts_x[curr] - pts_x[j], pts_y[curr] - pts_y[j]))
        visited.add(next_node)
        path.append(next_node)
        curr = next_node

    # If it was a closed loop (no degree-1 endpoints), append start to complete the 360-degree loop
    if not deg1:
        path.append(start)

    return pts_x[path], pts_y[path]


def extract_polar_ring_locus(
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    fom_2d: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts continuous closed ring loci using polar radial ray maximum-ridge tracing.
    Ideal for closed ring / annular degeneracy manifolds.
    """
    if config is None:
        config = {}
    locus_cfg = config.get("locus", {}) if isinstance(config.get("locus"), dict) else config
    n_sample = int(locus_cfg.get("sample_points", config.get("sample_points", 40)))
    smoothness = float(locus_cfg.get("smoothness", config.get("smoothness", 0.0001)))
    spline_deg = int(locus_cfg.get("spline_degree", config.get("spline_degree", 3)))
    center_cfg = locus_cfg.get("center", config.get("center", "auto"))

    x1_min, x1_max = float(grid_x1[0]), float(grid_x1[-1])
    x2_min, x2_max = float(grid_x2[0]), float(grid_x2[-1])
    n1, n2 = len(grid_x1), len(grid_x2)

    # 1. Determine ring center
    if isinstance(center_cfg, (list, tuple)) and len(center_cfg) == 2:
        x1c, x2c = float(center_cfg[0]), float(center_cfg[1])
    else:
        # Auto center: centroid of top 15% FOM points
        thresh_pct = float(locus_cfg.get("threshold_percentile", 85.0))
        finite_fom = fom_2d[np.isfinite(fom_2d)]
        if finite_fom.size == 0:
            return []
        cutoff = float(np.percentile(finite_fom, thresh_pct))
        mask = (fom_2d >= cutoff) & np.isfinite(fom_2d)
        y_idx, x_idx = np.where(mask)
        if len(x_idx) == 0:
            x1c = 0.5 * (x1_min + x1_max)
            x2c = 0.5 * (x2_min + x2_max)
        else:
            x1c = float(np.mean(grid_x1[x_idx]))
            x2c = float(np.mean(grid_x2[y_idx]))

    # Max radial reach
    max_r = min(x1c - x1_min, x1_max - x1c, x2c - x2_min, x2_max - x2c) * 0.95
    if max_r <= 0.005:
        max_r = min(x1_max - x1_min, x2_max - x2_min) * 0.45

    # 2. Polar rays
    thetas = np.linspace(0, 2 * np.pi, max(n_sample, 30), endpoint=False)
    r_scan = np.linspace(0.005, max_r, 250)

    ring_x1 = []
    ring_x2 = []
    for th in thetas:
        ux, uy = np.cos(th), np.sin(th)
        px = x1c + r_scan * ux
        py = x2c + r_scan * uy
        idx_x = (np.clip(px, x1_min, x1_max) - x1_min) / max(x1_max - x1_min, 1e-12) * (n1 - 1)
        idx_y = (np.clip(py, x2_min, x2_max) - x2_min) / max(x2_max - x2_min, 1e-12) * (n2 - 1)
        fom_ray = ndi.map_coordinates(fom_2d, [idx_y, idx_x], order=3, mode="nearest")
        best_r = r_scan[int(np.argmax(fom_ray))]
        ring_x1.append(float(x1c + best_r * ux))
        ring_x2.append(float(x2c + best_r * uy))

    # 3. Fit smooth periodic spline
    k = min(spline_deg, len(ring_x1) - 1, 3)
    try:
        tck, u = splprep([ring_x1, ring_x2], s=smoothness, k=k, per=True)
        u_fine = np.linspace(0, 1, n_sample, endpoint=False)
        curve_x1, curve_x2 = splev(u_fine, tck)
    except Exception:
        u_fine = np.linspace(0, 1, n_sample, endpoint=False)
        curve_x1 = np.interp(u_fine, np.linspace(0, 1, len(ring_x1)), ring_x1)
        curve_x2 = np.interp(u_fine, np.linspace(0, 1, len(ring_x2)), ring_x2)

    # Resample FOM along fitted closed curve
    idx_x = (np.clip(curve_x1, x1_min, x1_max) - x1_min) / max(x1_max - x1_min, 1e-12) * (n1 - 1)
    idx_y = (np.clip(curve_x2, x2_min, x2_max) - x2_min) / max(x2_max - x2_min, 1e-12) * (n2 - 1)
    fom_along_curve = ndi.map_coordinates(fom_2d, [idx_y, idx_x], order=3, mode="nearest")
    arc_length = float(np.sum(np.hypot(np.diff(np.r_[curve_x1, curve_x1[0]]), np.diff(np.r_[curve_x2, curve_x2[0]]))))

    return [{
        "locus_id": 1,
        "area_px": int(np.sum(fom_2d >= np.percentile(fom_2d[np.isfinite(fom_2d)], 60))),
        "max_fom": float(np.max(fom_along_curve)),
        "mean_fom": float(np.mean(fom_along_curve)),
        "length": arc_length,
        "is_closed": True,
        "center": [x1c, x2c],
        "r1": [float(v) for v in curve_x1],
        "r2": [float(v) for v in curve_x2],
        "fom": [float(v) for v in fom_along_curve],
    }]


def extract_optimal_loci(
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    fom_2d: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts continuous 1D optimal connected loci (degeneracy curves) from a 2D FOM landscape.
    Supports 'polar' (closed ring), 'cartesian' (open curve), and 'auto' geometry modes.
    """
    if config is None:
        config = {}

    locus_cfg = config.get("locus", {}) if isinstance(config.get("locus"), dict) else config
    mode = str(locus_cfg.get("mode", config.get("mode", locus_cfg.get("geometry_type", config.get("geometry_type", "auto"))))).lower()

    if mode in ["polar", "ring", "closed"]:
        return extract_polar_ring_locus(grid_x1, grid_x2, fom_2d, config=config)

    finite_fom = fom_2d[np.isfinite(fom_2d)]
    if finite_fom.size == 0:
        return []

    if mode == "auto":
        # Auto-detect annular ring topology (hole in threshold mask)
        thresh_test = float(locus_cfg.get("threshold_percentile", 80.0))
        cutoff_test = float(np.percentile(finite_fom, thresh_test))
        mask_test = (fom_2d >= cutoff_test) & np.isfinite(fom_2d)
        filled_test = ndi.binary_fill_holes(mask_test)
        if np.sum(filled_test) > (np.sum(mask_test) + 15):
            return extract_polar_ring_locus(grid_x1, grid_x2, fom_2d, config=config)

    thresh_pct = float(locus_cfg.get("threshold_percentile", config.get("threshold_percentile", 90.0)))
    min_area = int(locus_cfg.get("min_locus_area_px", config.get("min_locus_area_px", 25)))
    max_loci = int(locus_cfg.get("max_loci", config.get("max_loci", config.get("number_of_loci", 1))))
    smoothness = float(locus_cfg.get("smoothness", config.get("smoothness", 0.001)))
    spline_degree = int(locus_cfg.get("spline_degree", config.get("spline_degree", 3)))
    n_sample = int(locus_cfg.get("sample_points", config.get("sample_points", 50)))

    cutoff = float(np.percentile(finite_fom, thresh_pct))
    mask = (fom_2d >= cutoff) & np.isfinite(fom_2d)
    mask = ndi.binary_closing(mask)

    labeled, num_features = ndi.label(mask)
    if num_features == 0:
        return []

    # Rank connected components by maximum FOM value and area
    components = []
    for lbl in range(1, num_features + 1):
        comp_mask = (labeled == lbl)
        area = int(np.sum(comp_mask))
        if area < min_area:
            continue
        max_val = float(np.max(fom_2d[comp_mask]))
        mean_val = float(np.mean(fom_2d[comp_mask]))
        components.append((lbl, area, max_val, mean_val, comp_mask))

    if not components:
        return []

    # Sort descending by max FOM
    components.sort(key=lambda c: c[2], reverse=True)
    selected_components = components[:max_loci]

    loci_results = []
    for locus_idx, (lbl, area, max_val, mean_val, comp_mask) in enumerate(selected_components, start=1):
        skel = skeletonize_mask(comp_mask)
        y_idx, x_idx = np.where(skel)

        if len(x_idx) < 3:
            continue

        raw_x1 = grid_x1[x_idx]
        raw_x2 = grid_x2[y_idx]

        # Order skeleton pixels along contiguous path
        ord_x1, ord_x2 = order_skeleton_points(raw_x1, raw_x2)

        # Remove duplicate adjacent points
        dists = np.hypot(np.diff(ord_x1), np.diff(ord_x2))
        valid_steps = np.r_[True, dists > 1e-8]
        ord_x1 = ord_x1[valid_steps]
        ord_x2 = ord_x2[valid_steps]

        is_closed_loop = (len(ord_x1) >= 6 and np.hypot(ord_x1[0] - ord_x1[-1], ord_x2[0] - ord_x2[-1]) < 0.02)

        if len(ord_x1) < 4:
            # Fallback to linear interpolation if too few points for cubic spline
            u_fine = np.linspace(0, 1, n_sample)
            curve_x1 = np.interp(u_fine, np.linspace(0, 1, len(ord_x1)), ord_x1)
            curve_x2 = np.interp(u_fine, np.linspace(0, 1, len(ord_x2)), ord_x2)
        else:
            k = min(spline_degree, len(ord_x1) - 1, 3)
            try:
                if is_closed_loop:
                    tck, u = splprep([ord_x1[:-1], ord_x2[:-1]], s=smoothness, k=k, per=True)
                else:
                    tck, u = splprep([ord_x1, ord_x2], s=smoothness, k=k, per=False)
                u_fine = np.linspace(0, 1, n_sample)
                curve_x1, curve_x2 = splev(u_fine, tck)
            except Exception:
                # Fallback to linear interpolation
                u_fine = np.linspace(0, 1, n_sample)
                curve_x1 = np.interp(u_fine, np.linspace(0, 1, len(ord_x1)), ord_x1)
                curve_x2 = np.interp(u_fine, np.linspace(0, 1, len(ord_x2)), ord_x2)

        # Sample FOM along the fitted curve using bicubic interpolation
        # Map physical coordinates to grid indices
        x1_min, x1_max = float(grid_x1[0]), float(grid_x1[-1])
        x2_min, x2_max = float(grid_x2[0]), float(grid_x2[-1])
        n1, n2 = len(grid_x1), len(grid_x2)

        idx_x = (np.clip(curve_x1, x1_min, x1_max) - x1_min) / max(x1_max - x1_min, 1e-12) * (n1 - 1)
        idx_y = (np.clip(curve_x2, x2_min, x2_max) - x2_min) / max(x2_max - x2_min, 1e-12) * (n2 - 1)
        fom_along_curve = ndi.map_coordinates(fom_2d, [idx_y, idx_x], order=3, mode="nearest")

        arc_length = float(np.sum(np.hypot(np.diff(curve_x1), np.diff(curve_x2))))

        loci_results.append({
            "locus_id": locus_idx,
            "area_px": area,
            "max_fom": max_val,
            "mean_fom": mean_val,
            "length": arc_length,
            "r1": [float(v) for v in curve_x1],
            "r2": [float(v) for v in curve_x2],
            "fom": [float(v) for v in fom_along_curve],
        })

    return loci_results


def compute_curve_normals(r1_pts: np.ndarray, r2_pts: np.ndarray) -> np.ndarray:
    """
    Computes 2D unit normal vectors perpendicular to each point along the curve.
    For tangent vector t = (dx, dy), normal vector is n = (-dy, dx) normalized to unit length.
    """
    r1_arr = np.asarray(r1_pts, dtype=float)
    r2_arr = np.asarray(r2_pts, dtype=float)
    n = len(r1_arr)
    normals = np.zeros((n, 2), dtype=float)
    if n < 2:
        return np.array([[0.0, 1.0]] * n)

    # Compute tangents using central differences (and one-sided differences at endpoints)
    dx = np.zeros(n)
    dy = np.zeros(n)

    dx[0] = r1_arr[1] - r1_arr[0]
    dy[0] = r2_arr[1] - r2_arr[0]

    dx[-1] = r1_arr[-1] - r1_arr[-2]
    dy[-1] = r2_arr[-1] - r2_arr[-2]

    if n > 2:
        dx[1:-1] = (r1_arr[2:] - r1_arr[:-2]) / 2.0
        dy[1:-1] = (r2_arr[2:] - r2_arr[:-2]) / 2.0

    lengths = np.hypot(dx, dy)
    lengths[lengths < 1e-12] = 1.0

    tx = dx / lengths
    ty = dy / lengths

    # Unit normal rotated 90 degrees CCW: (-ty, tx)
    normals[:, 0] = -ty
    normals[:, 1] = tx

    return normals


def export_loci_to_csv(loci: List[Dict[str, Any]], csv_path: Union[str, Path]) -> None:
    """
    Saves extracted loci to a structured CSV file.
    Includes group_velocity, residual_gap, and is_valid status columns if present in locus records.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    has_vg = any("group_velocity" in l or "vg" in l for l in loci)
    has_gap = any("residual_gap" in l or "gaps" in l for l in loci)
    has_unrefined = any("r1_unrefined" in l for l in loci)
    has_valid = any("is_valid" in l for l in loci)

    headers = ["locus_id", "point_index", "t_normalized", "r1", "r2"]
    if has_unrefined:
        headers.extend(["r1_initial", "r2_initial"])
    headers.append("predicted_FOM")
    if has_gap:
        headers.append("residual_gap")
    if has_vg:
        headers.append("group_velocity")
    if has_valid:
        headers.append("is_valid")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for locus in loci:
            l_id = locus["locus_id"]
            r1_vals = locus["r1"]
            r2_vals = locus["r2"]
            r1_unref = locus.get("r1_unrefined", [])
            r2_unref = locus.get("r2_unrefined", [])
            fom_vals = locus.get("fom", [])
            gap_vals = locus.get("residual_gap") or locus.get("gaps") or []
            vg_vals = locus.get("group_velocity") or locus.get("vg") or []
            valid_vals = locus.get("is_valid", [])
            n = len(r1_vals)
            for idx in range(n):
                t = float(idx) / max(n - 1, 1)
                row = [l_id, idx + 1, f"{t:.4f}", f"{r1_vals[idx]:.6f}", f"{r2_vals[idx]:.6f}"]
                if has_unrefined:
                    if idx < len(r1_unref) and idx < len(r2_unref):
                        row.extend([f"{r1_unref[idx]:.6f}", f"{r2_unref[idx]:.6f}"])
                    else:
                        row.extend(["nan", "nan"])
                fom_v = fom_vals[idx] if idx < len(fom_vals) else 0.0
                row.append(f"{fom_v:.6e}")
                if has_gap:
                    if idx < len(gap_vals) and gap_vals[idx] is not None:
                        row.append(f"{float(gap_vals[idx]):.6e}")
                    else:
                        row.append("nan")
                if has_vg:
                    if idx < len(vg_vals) and vg_vals[idx] is not None:
                        row.append(f"{float(vg_vals[idx]):.6f}")
                    else:
                        row.append("nan")
                if has_valid:
                    if idx < len(valid_vals):
                        row.append(str(bool(valid_vals[idx])))
                    else:
                        row.append("True")
                writer.writerow(row)


def export_loci_to_json(loci: List[Dict[str, Any]], json_path: Union[str, Path]) -> None:
    """
    Saves extracted loci metadata and points to structured JSON.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"num_loci": len(loci), "loci": loci}, f, indent=2)


def plot_locus_profiles(
    loci: List[Dict[str, Any]],
    param_names: Optional[List[str]] = None,
    param_bounds: Optional[List[Tuple[float, float]]] = None,
    output_path: Union[str, Path] = "bo_locus_profile.png",
    title: str = "Optimal Locus Analysis",
) -> None:
    """
    Plots a 3-panel figure analyzing the extracted optimal locus:
    - Panel (a): Parameter Space trajectory (r1 vs r2) color-coded by group velocity vg.
    - Panel (b): Group Velocity profile vg/c vs normalized arc length t.
    - Panel (c): Figure of Merit profile FOM vs normalized arc length t (log scale).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FormatStrFormatter

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not loci:
        return

    p1_name = param_names[0] if param_names and len(param_names) > 0 else "r1"
    p2_name = param_names[1] if param_names and len(param_names) > 1 else "r2"

    p1_label = f"${p1_name[0]}_{{{p1_name[1:]}}}/a$" if len(p1_name) > 1 and p1_name[1:].isdigit() else f"${p1_name}/a$"
    p2_label = f"${p2_name[0]}_{{{p2_name[1:]}}}/a$" if len(p2_name) > 1 and p2_name[1:].isdigit() else f"${p2_name}/a$"

    has_vg = any("group_velocity" in l or "vg" in l for l in loci)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # 3-panel horizontal layout
    fig = plt.figure(figsize=(16.2, 4.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.38)

    ax_param = fig.add_subplot(gs[0, 0])
    ax_vg = fig.add_subplot(gs[0, 1])
    ax_fom = fig.add_subplot(gs[0, 2])

    colors = ["#00B050", "#0070C0", "#E30613", "#7030A0"]

    for idx, locus in enumerate(loci):
        c = colors[idx % len(colors)]
        l_id = locus["locus_id"]
        r1_vals = np.array(locus["r1"])
        r2_vals = np.array(locus["r2"])
        fom_vals = np.array(locus["fom"])
        t_vals = np.linspace(0, 1, len(r1_vals))
        vg_vals = np.array(locus.get("group_velocity") or locus.get("vg") or [])

        lbl = f"Locus #{l_id}" if len(loci) > 1 else "Optimal Locus"

        # -------------------------------------------------------------
        # Panel (a): Parameter Space Trajectory
        # -------------------------------------------------------------
        if has_vg and len(vg_vals) == len(r1_vals):
            # Connect curve with thin underlying line
            ax_param.plot(r1_vals, r2_vals, color="#888888", linestyle="--", linewidth=1.5, zorder=3)
            # Scatter color-coded by group velocity
            sc = ax_param.scatter(
                r1_vals,
                r2_vals,
                c=vg_vals,
                cmap="plasma",
                s=40,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
                label=lbl,
            )
            divider = make_axes_locatable(ax_param)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cbar_p = fig.colorbar(sc, cax=cax)
            cbar_p.set_label(r"$v_g / c$", fontsize=10, fontweight="bold")
            cbar_p.ax.tick_params(labelsize=8.5)
        else:
            ax_param.plot(r1_vals, r2_vals, "-o", color=c, linewidth=2.0, markersize=4.0, zorder=4, label=lbl)

        # Mark Start (t=0) and End (t=1)
        ax_param.scatter(r1_vals[0], r2_vals[0], c="#00FF00", edgecolors="black", marker="o", s=85, zorder=6, label=r"Start ($t=0$)")
        ax_param.scatter(r1_vals[-1], r2_vals[-1], c="#FF0000", edgecolors="black", marker="s", s=85, zorder=6, label=r"End ($t=1$)")

        ax_param.set_xlabel(p1_label, fontsize=11, fontweight="bold")
        ax_param.set_ylabel(p2_label, fontsize=11, fontweight="bold")
        ax_param.set_title("(a) Parameter Space Trajectory", fontsize=11.5, fontweight="bold")
        ax_param.set_aspect("equal", adjustable="box")
        ax_param.grid(True, linestyle=":", alpha=0.6)
        ax_param.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax_param.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax_param.legend(loc="center", frameon=True, framealpha=0.85, fontsize=8.5)

        if param_bounds and len(param_bounds) >= 2:
            ax_param.set_xlim(float(param_bounds[0][0]), float(param_bounds[0][1]))
            ax_param.set_ylim(float(param_bounds[1][0]), float(param_bounds[1][1]))

        # -------------------------------------------------------------
        # Panel (b): Group Velocity Profile
        # -------------------------------------------------------------
        if has_vg and len(vg_vals) == len(t_vals):
            ax_vg.plot(t_vals, vg_vals, "-o", color="#0070C0", linewidth=2.0, markersize=4.5, label=r"$v_g(t)$")
            ax_vg.set_ylabel(r"Group Velocity $v_g / c$", fontsize=11, fontweight="bold")
            ax_vg.set_xlabel(r"Normalized Trajectory $t \in [0, 1]$", fontsize=11, fontweight="bold")
            ax_vg.set_title(r"(b) Group Velocity $v_g / c$", fontsize=11.5, fontweight="bold")
            ax_vg.grid(True, linestyle=":", alpha=0.6)
            ax_vg.legend(loc="upper right", frameon=True, fontsize=9)
        else:
            ax_vg.text(0.5, 0.5, "Group Velocity\nNot Evaluated", horizontalalignment="center", verticalalignment="center", transform=ax_vg.transAxes)
            ax_vg.set_title(r"(b) Group Velocity $v_g / c$", fontsize=11.5, fontweight="bold")

        # -------------------------------------------------------------
        # Panel (c): Figure of Merit Profile
        # -------------------------------------------------------------
        ax_fom.plot(t_vals, fom_vals, "-s", color="#00B050", linewidth=1.8, markersize=4.0, label=r"$\mathrm{FOM}(t)$")
        ax_fom.set_yscale("log")
        ax_fom.set_ylabel(r"$\mathrm{FOM} = \mathbb{E}[C]^{-1}$", fontsize=11, fontweight="bold")
        ax_fom.set_xlabel(r"Normalized Trajectory $t \in [0, 1]$", fontsize=11, fontweight="bold")
        ax_fom.set_title(r"(c) $\mathrm{FOM} = \mathbb{E}[C]^{-1}$", fontsize=11.5, fontweight="bold")
        ax_fom.grid(True, linestyle=":", alpha=0.6)
        ax_fom.legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle(f"{title}", fontsize=13, fontweight="bold", y=0.99)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved locus profile plot to '{path}'")
