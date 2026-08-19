"""
Common Photonic Crystal Simulation and Geometry Utilities
Shared across Bayesian Optimization, Workflow pipelines, Plotters, and Extractors.
"""

import os
import re
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List, Any
import numpy as np
import h5py
from scipy.ndimage import label, binary_opening


# =============================================================================
# 1. Dielectric Geometry Connectivity & Topology Checking
# =============================================================================

def check_array_connectivity(
    epsilon: np.ndarray,
    epsilon_threshold: float = 1.1,
    check_pbc: bool = True,
    min_neck_width_px: int = 1
) -> Tuple[bool, int, str]:
    """
    Checks if the high-dielectric region (epsilon > epsilon_threshold) in a 2D or 3D numpy array
    forms a continuous, unbroken connected slab spanning across the unit cell.
    """
    mask = (epsilon > epsilon_threshold)

    if not np.any(mask):
        return False, 0, "No dielectric material above threshold."

    if np.all(mask):
        return True, 1, "Uniform dielectric slab."

    if check_pbc:
        if mask.ndim == 2:
            tiled_mask = np.tile(mask, (3, 3))
        elif mask.ndim == 3:
            tiled_mask = np.tile(mask, (3, 3, 1))
        else:
            tiled_mask = mask
    else:
        tiled_mask = mask

    if min_neck_width_px > 1:
        if mask.ndim == 2:
            structure = np.ones((min_neck_width_px, min_neck_width_px), dtype=bool)
        else:
            structure = np.ones((min_neck_width_px, min_neck_width_px, 1), dtype=bool)
        tiled_mask = binary_opening(tiled_mask, structure=structure)
        if not np.any(tiled_mask):
            return False, 0, f"Dielectric connections severed after opening (neck < {min_neck_width_px} px)."

    labeled_array, num_features = label(tiled_mask)

    if num_features == 0:
        return False, 0, "No connected dielectric component found."

    # Verify span across periodic boundaries
    if check_pbc:
        if mask.ndim == 2:
            ny, nx = mask.shape
            center_labels = set(np.unique(labeled_array[ny:2*ny, nx:2*nx])) - {0}
            left_labels = set(np.unique(labeled_array[ny:2*ny, :nx])) - {0}
            right_labels = set(np.unique(labeled_array[ny:2*ny, 2*nx:])) - {0}
            bottom_labels = set(np.unique(labeled_array[:ny, nx:2*nx])) - {0}
            top_labels = set(np.unique(labeled_array[2*ny:, nx:2*nx])) - {0}

            spans_x = bool(center_labels & left_labels & right_labels)
            spans_y = bool(center_labels & bottom_labels & top_labels)

            if spans_x and spans_y:
                return True, num_features, "Continuous dielectric matrix spanning both x and y boundaries."
            else:
                return False, num_features, f"Disconnected dielectric (spans_x={spans_x}, spans_y={spans_y})."

        elif mask.ndim == 3:
            ny, nx, nz = mask.shape
            center_labels = set(np.unique(labeled_array[ny:2*ny, nx:2*nx, :])) - {0}
            left_labels = set(np.unique(labeled_array[ny:2*ny, :nx, :])) - {0}
            right_labels = set(np.unique(labeled_array[ny:2*ny, 2*nx:, :])) - {0}
            bottom_labels = set(np.unique(labeled_array[:ny, nx:2*nx, :])) - {0}
            top_labels = set(np.unique(labeled_array[2*ny:, nx:2*nx, :])) - {0}

            spans_x = bool(center_labels & left_labels & right_labels)
            spans_y = bool(center_labels & bottom_labels & top_labels)

            if spans_x and spans_y:
                return True, num_features, "Continuous dielectric slab spanning across x and y boundaries."
            else:
                return False, num_features, f"Disconnected slab (spans_x={spans_x}, spans_y={spans_y})."

    return (num_features == 1), num_features, f"Dielectric components: {num_features}"


def check_slab_connectivity(
    h5_file_path: Union[str, Path],
    epsilon_threshold: float = 1.1,
    check_pbc: bool = True,
    min_neck_width_px: int = 1
) -> Tuple[bool, str]:
    """
    Loads epsilon from an MPB HDF5 file and checks if dielectric matrix is continuous.
    """
    path = Path(h5_file_path).resolve()
    if not path.is_file():
        return False, f"File not found: '{path}'"

    try:
        with h5py.File(path, "r") as f:
            if "data" in f:
                eps_data = f["data"][()]
            elif "epsilon" in f:
                eps_data = f["epsilon"][()]
            else:
                for k in f.keys():
                    if "eps" in k.lower() or "data" in k.lower():
                        eps_data = f[k][()]
                        break
                else:
                    return False, f"No dielectric dataset found in '{path.name}'."

        is_conn, n_comp, msg = check_array_connectivity(
            eps_data,
            epsilon_threshold=epsilon_threshold,
            check_pbc=check_pbc,
            min_neck_width_px=min_neck_width_px
        )
        return is_conn, msg
    except Exception as e:
        return False, f"Error processing '{path.name}': {e}"


# =============================================================================
# 2. 2D/3D Parity Aliasing & Matcher
# =============================================================================

def is_parity_match(pol1: str, pol2: str) -> bool:
    """
    Checks if two polarization / parity strings match, including 2D/3D aliases:
    'zeven' <-> 'te', 'zodd' <-> 'tm'.
    """
    p1 = str(pol1).strip().lower()
    p2 = str(pol2).strip().lower()
    if p1 == p2:
        return True
    if (p1 == "zeven" and p2 == "te") or (p1 == "te" and p2 == "zeven"):
        return True
    if (p1 == "zodd" and p2 == "tm") or (p1 == "tm" and p2 == "zodd"):
        return True
    return False


# =============================================================================
# 3. Safe Output Log Reader & Extractor
# =============================================================================

def read_mpb_output_log(directory: Union[str, Path]) -> str:
    """
    Safely reads the MPB output log text from either directory/output/output.out
    or directory/output.out.
    """
    d = Path(directory).resolve()
    candidates = [
        d / "output" / "output.out",
        d / "output.out",
    ]
    for c in candidates:
        if c.is_file():
            try:
                return c.read_text()
            except Exception:
                pass
    return ""


def extract_gamma_frequencies(output_text: str, pol: str = "te") -> Dict[int, float]:
    """
    Extracts Gamma point frequencies from MPB output text.
    """
    freq_dict = {}
    lines = output_text.splitlines()
    for line in lines:
        if (f"{pol}freqs:" in line or "tefreqs:" in line or "zevenfreqs:" in line) and "k index" not in line:
            parts = [p.strip() for p in line.split(":")[-1].split(",") if p.strip()]
            if len(parts) >= 6:
                for b_idx, f_str in enumerate(parts[5:], start=1):
                    try:
                        freq_dict[b_idx] = float(f_str)
                    except ValueError:
                        pass
    return freq_dict
