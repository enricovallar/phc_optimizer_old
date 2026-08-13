"""
Dielectric Geometry Connectivity & Topology Checking Engine
"""

import os
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import h5py
from scipy.ndimage import label, binary_opening


def check_array_connectivity(
    epsilon: np.ndarray,
    epsilon_threshold: float = 1.1,
    check_pbc: bool = True,
    min_neck_width_px: int = 1
) -> Tuple[bool, int, str]:
    """
    Checks if the high-dielectric region (epsilon > epsilon_threshold) in a 2D or 3D numpy array
    forms a continuous, unbroken connected slab spanning across the unit cell.

    Parameters:
    -----------
    epsilon : np.ndarray
        2D or 3D dielectric grid array.
    epsilon_threshold : float
        Dielectric constant threshold to distinguish matrix/slab (default: 1.1).
    check_pbc : bool
        If True, tiles the unit cell 3x3 to account for periodic boundary conditions.
    min_neck_width_px : int
        Minimum feature/neck width threshold in grid pixels. Connections narrower than
        this pixel width are severed via morphological opening and flagged as invalid (default: 1).

    Returns:
    --------
    Tuple[bool, int, str]:
        (is_connected, num_components, status_message)
    """
    mask = (epsilon > epsilon_threshold)

    if not np.any(mask):
        return False, 0, "No dielectric material above threshold."

    # If all grid points are dielectric, it's fully connected
    if np.all(mask):
        return True, 1, "Uniform dielectric slab."

    if check_pbc:
        # Tile 3x3 in 2D (or along x, y for 3D) to handle periodic boundary conditions
        if mask.ndim == 2:
            tiled_mask = np.tile(mask, (3, 3))
        elif mask.ndim == 3:
            tiled_mask = np.tile(mask, (3, 3, 1))
        else:
            tiled_mask = mask
    else:
        tiled_mask = mask

    # If min_neck_width_px > 1, perform morphological opening to sever thin connections/necks
    if min_neck_width_px > 1:
        r = max(1, int(np.ceil(min_neck_width_px / 2.0)))
        if mask.ndim == 2:
            y_k, x_k = np.ogrid[-r:r+1, -r:r+1]
            struct_elem = (x_k**2 + y_k**2 <= r**2)
        else:
            y_k, x_k, _ = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
            struct_elem = (x_k**2 + y_k**2 <= r**2)
        opened_mask = binary_opening(tiled_mask, structure=struct_elem)
    else:
        opened_mask = tiled_mask

    labeled_grid, num_features = label(opened_mask)

    if num_features == 0:
        return False, 0, "No connected components found."

    if check_pbc and mask.ndim in (2, 3):
        ny, nx = mask.shape[0], mask.shape[1]
        # Central unit cell coordinates in 3x3 tiled grid
        c_y_min, c_y_max = ny, 2 * ny
        c_x_min, c_x_max = nx, 2 * nx

        # Get all component labels present in the central unit cell
        if mask.ndim == 2:
            central_labels = set(np.unique(labeled_grid[c_y_min:c_y_max, c_x_min:c_x_max]))
        else:
            central_labels = set(np.unique(labeled_grid[c_y_min:c_y_max, c_x_min:c_x_max, :]))

        central_labels.discard(0)  # Remove background label

        if not central_labels:
            return False, 0, "Central unit cell has no dielectric material."

        # Check if at least one central component spans across x (left to right) and y (bottom to top)
        spans_x = False
        spans_y = False

        for lbl in central_labels:
            component_coords = np.where(labeled_grid == lbl)
            y_coords, x_coords = component_coords[0], component_coords[1]

            has_left = np.any(x_coords < nx)
            has_right = np.any(x_coords >= 2 * nx)
            has_bottom = np.any(y_coords < ny)
            has_top = np.any(y_coords >= 2 * ny)

            if has_left and has_right:
                spans_x = True
            if has_bottom and has_top:
                spans_y = True

            if spans_x and spans_y:
                msg = "Slab is continuous and unbroken across unit cell boundaries."
                if min_neck_width_px > 1:
                    msg += f" (passes >={min_neck_width_px}px neck width constraint)"
                return True, num_features, msg

        if not (spans_x and spans_y):
            missing = []
            if not spans_x:
                missing.append("X-axis (left-right)")
            if not spans_y:
                missing.append("Y-axis (top-bottom)")
            reason = f"Slab breaks unit cell continuity along {', '.join(missing)}"
            if min_neck_width_px > 1:
                reason += f" (connection is too narrow, <{min_neck_width_px}px width)"
            return False, num_features, reason

    return True, num_features, f"Slab is connected ({num_features} components)."


def check_slab_connectivity(
    h5_path: Union[str, Path],
    epsilon_threshold: float = 1.1,
    check_pbc: bool = True,
    min_neck_width_px: int = 1
) -> Tuple[bool, int, str]:
    """
    Reads an MPB dielectric HDF5 file (e.g. main-epsilon.h5) and verifies whether
    the dielectric slab is a connected region and satisfies minimum neck width.
    """
    h5_file = Path(h5_path)
    if not h5_file.is_file():
        return False, 0, f"HDF5 file not found: '{h5_file}'"

    try:
        with h5py.File(h5_file, "r") as f:
            key = None
            for k in ["epsilon", "data"]:
                if k in f:
                    key = k
                    break
            if key is None:
                key = list(f.keys())[0]

            epsilon = f[key][()]
    except Exception as e:
        return False, 0, f"Failed to read HDF5 dielectric grid: {e}"

    return check_array_connectivity(
        epsilon,
        epsilon_threshold=epsilon_threshold,
        check_pbc=check_pbc,
        min_neck_width_px=min_neck_width_px
    )
