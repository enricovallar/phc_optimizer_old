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
    Orders disordered 2D skeleton pixel coordinates into a continuous sequential trajectory
    from one endpoint to the other using greedy nearest-neighbor traversal.
    """
    if len(pts_x) <= 2:
        return pts_x, pts_y

    pts = np.c_[pts_x, pts_y]
    n = len(pts)

    # Find the extremal starting point (furthest from centroid along principal axis)
    centroid = np.mean(pts, axis=0)
    dists_from_centroid = np.linalg.norm(pts - centroid, axis=1)
    start_idx = int(np.argmax(dists_from_centroid))

    unvisited = set(range(n))
    ordered_indices = [start_idx]
    unvisited.remove(start_idx)

    curr = start_idx
    while unvisited:
        curr_pt = pts[curr]
        rem_indices = list(unvisited)
        rem_pts = pts[rem_indices]
        dists = np.linalg.norm(rem_pts - curr_pt, axis=1)
        nearest_pos = int(np.argmin(dists))
        next_idx = rem_indices[nearest_pos]

        ordered_indices.append(next_idx)
        unvisited.remove(next_idx)
        curr = next_idx

    ordered_x = pts_x[ordered_indices]
    ordered_y = pts_y[ordered_indices]
    return ordered_x, ordered_y


def extract_optimal_loci(
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    fom_2d: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts continuous 1D optimal connected loci (degeneracy curves) from a 2D FOM landscape.

    Parameters:
    -----------
    grid_x1 : np.ndarray
        1D array of parameter 1 coordinates (shape: N1,).
    grid_x2 : np.ndarray
        1D array of parameter 2 coordinates (shape: N2,).
    fom_2d : np.ndarray
        2D scalar field of FOM values (shape: N2, N1).
    config : dict, optional
        Postprocessing configuration options:
        - threshold_percentile : float (default: 90.0)
        - min_locus_area_px : int (default: 25)
        - max_loci : int (default: 1)
        - smoothness : float (default: 0.001)
        - spline_degree : int (default: 3)
        - sample_points : int (default: 50)

    Returns:
    --------
    List[dict]:
        List of extracted loci, each containing:
        - 'locus_id': int (1, 2, ...)
        - 'area_px': int
        - 'max_fom': float
        - 'mean_fom': float
        - 'r1': List[float] (resampled parameter 1 coordinates)
        - 'r2': List[float] (resampled parameter 2 coordinates)
        - 'fom': List[float] (interpolated FOM values along the curve)
        - 'length': float (approximate arc length in parameter space)
    """
    if config is None:
        config = {}

    thresh_pct = float(config.get("threshold_percentile", 90.0))
    min_area = int(config.get("min_locus_area_px", 25))
    max_loci = int(config.get("max_loci", config.get("number_of_loci", 1)))
    smoothness = float(config.get("smoothness", 0.001))
    spline_degree = int(config.get("spline_degree", 3))
    n_sample = int(config.get("sample_points", 50))

    finite_fom = fom_2d[np.isfinite(fom_2d)]
    if finite_fom.size == 0:
        return []

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

        if len(ord_x1) < 4:
            # Fallback to linear interpolation if too few points for cubic spline
            u_fine = np.linspace(0, 1, n_sample)
            curve_x1 = np.interp(u_fine, np.linspace(0, 1, len(ord_x1)), ord_x1)
            curve_x2 = np.interp(u_fine, np.linspace(0, 1, len(ord_x2)), ord_x2)
        else:
            k = min(spline_degree, len(ord_x1) - 1, 3)
            try:
                tck, u = splprep([ord_x1, ord_x2], s=smoothness, k=k)
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


def export_loci_to_csv(loci: List[Dict[str, Any]], csv_path: Union[str, Path]) -> None:
    """
    Saves extracted loci to a structured CSV file.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["locus_id", "point_index", "t_normalized", "r1", "r2", "predicted_FOM"])
        for locus in loci:
            l_id = locus["locus_id"]
            r1_vals = locus["r1"]
            r2_vals = locus["r2"]
            fom_vals = locus["fom"]
            n = len(r1_vals)
            for idx in range(n):
                t = float(idx) / max(n - 1, 1)
                writer.writerow([l_id, idx + 1, f"{t:.4f}", f"{r1_vals[idx]:.6f}", f"{r2_vals[idx]:.6f}", f"{fom_vals[idx]:.6e}"])


def export_loci_to_json(loci: List[Dict[str, Any]], json_path: Union[str, Path]) -> None:
    """
    Saves extracted loci metadata and points to structured JSON.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"num_loci": len(loci), "loci": loci}, f, indent=2)
