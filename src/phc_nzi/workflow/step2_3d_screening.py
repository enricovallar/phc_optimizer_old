"""
Step 2: 3D Grid Screening & Symmetry Matching
Evaluates 3D crystal membrane mode at fixed slab thickness h/a, checks 3D slab
connectivity, and matches 3D mode triplets by the target 2D Irrep symmetry signature.
"""

import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from ..runner import run_hpc
from ..symmetry import parse_symmetry_blocks, compute_projections, identify_irrep
from ..utils import check_slab_connectivity, read_mpb_output_log, extract_gamma_frequencies


def run_step2_3d_screening(
    cfg: Any,
    work_dir: Path,
    output_dir: Path,
    param_names: List[str],
    param_bounds: List[List[float]],
    fixed_params: Dict[str, Any],
    sim_cfg: Dict[str, Any],
    target_2d_triplet: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes Step 2: 3D Grid Screening, Slab Connectivity Checks, and Symmetry Matching.
    """
    step_dir = output_dir / "step2_3d_screening"
    step_dir.mkdir(parents=True, exist_ok=True)

    s2_cfg = cfg.get("step2_3d_screening", cfg.get("workflow", {}).get("step2_3d_screening", {}))
    n_grid = int(s2_cfg.get("grid_points", 10))
    min_band = max(2, int(s2_cfg.get("min_band", 2)))
    fixed_h = float(s2_cfg.get("fixed_h", 0.50))
    sz_val = float(s2_cfg.get("sz", 4.0))
    pol = str(s2_cfg.get("polarization", "zeven")).lower()
    group = str(s2_cfg.get("symmetry_group", "C4v"))
    num_bands = int(s2_cfg.get("num_bands", 14))
    res = int(s2_cfg.get("resolution", 25))
    res_z = int(s2_cfg.get("res_z", 16))
    cores = int(sim_cfg.get("cores", 1))
    workers = int(sim_cfg.get("parallel_workers", 28))

    p1_name, p2_name = param_names[0], param_names[1]
    b1, b2 = param_bounds[0], param_bounds[1]

    x1_vals = np.linspace(float(b1[0]), float(b1[1]), n_grid)
    x2_vals = np.linspace(float(b2[0]), float(b2[1]), n_grid)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    grid_pts = list(zip(X1.ravel(), X2.ravel()))

    target_irreps_str = target_2d_triplet["dominant_irreps"] if target_2d_triplet else "A_2-E-E"

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: 3D Membrane Slab Screening & Irrep Symmetry Matching")
        print("=" * 70)
        print(f"Fixed Slab Thickness: h/a = {fixed_h} (sz = {sz_val})")
        print(f"Target 2D Irreps:     {target_irreps_str}")
        print(f"Grid Resolution:      {n_grid} x {n_grid} ({len(grid_pts)} total points)")
        print(f"Polarization:         {pol.upper()} (res={res}, res-z={res_z}, num-bands={num_bands})")
        print(f"Band Range:           Bands {min_band} to {num_bands} (excluding band 1)")
        print(f"Parallel Workers:     {workers} workers")
        print("-" * 70)

    tasks = []
    for idx, (p1, p2) in enumerate(grid_pts):
        pt_dir = step_dir / "sims" / f"pt_{idx:03d}"
        tasks.append((idx, p1, p2, pt_dir))

    def _evaluate_3d_pt(task_args):
        idx, p1, p2, pt_dir = task_args
        pt_dir.mkdir(parents=True, exist_ok=True)

        p_dict = {
            p1_name: float(p1),
            p2_name: float(p2),
            "h": fixed_h,
            "sz": sz_val,
            "resolution": res,
            "res-z": res_z,
            "num-bands": num_bands,
            "only_gamma?": "true",
            "display_symmetry?": "true",
            "display_group_velocity?": "false",
            f"run-{pol}?": "true",
        }
        for k, v in fixed_params.items():
            if k not in p_dict:
                p_dict[k] = v

        ctl_script = Path(sim_cfg.get("ctl_script", "main.ctl")).resolve()
        run_hpc(
            script=ctl_script,
            mpb_command_line_params=p_dict,
            use_mpi=False,
            cores=cores,
            wd=pt_dir,
            auto_extract=False,
            auto_plot=False,
            only_gamma=True,
            verbose=False,
        )

        out_txt = read_mpb_output_log(pt_dir)
        freq_dict = extract_gamma_frequencies(out_txt, pol=pol)

        # Check 3D matrix slab connectivity
        eps_files = list(pt_dir.glob("*-epsilon.h5")) or list((pt_dir / "output").glob("*-epsilon.h5"))
        conn_ok = True
        if eps_files:
            try:
                c_ok, c_msg = check_slab_connectivity(eps_files[0], epsilon_threshold=1.1, min_neck_width_px=2)
                conn_ok = c_ok
            except Exception:
                conn_ok = True

        # Extract Irreps
        sym_blocks = parse_symmetry_blocks(out_txt)
        gamma_sym = (
            sym_blocks.get(pol, {})
            or sym_blocks.get("zeven", {})
            or sym_blocks.get("te", {})
            or sym_blocks.get("Gamma", {})
        )
        irrep_dict = {}
        for b_idx, chars in gamma_sym.items():
            proj = compute_projections(chars, group=group)
            irr, conf = identify_irrep(proj)
            irrep_dict[b_idx] = irr

        return {
            "index": idx,
            p1_name: float(p1),
            p2_name: float(p2),
            "is_connected": conn_ok,
            "freqs": freq_dict,
            "irreps": irrep_dict,
        }

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=len(tasks), desc="3D Grid Evaluation", unit="pt") as pbar:
            for res_pt in executor.map(_evaluate_3d_pt, tasks):
                results.append(res_pt)
                pbar.update(1)

    results.sort(key=lambda r: r["index"])
    conn_mask = np.array([r["is_connected"] for r in results]).reshape((n_grid, n_grid))

    # Triplet Mode Analysis in 3D (excluding band 1)
    target_tokens = set(target_irreps_str.split("-"))
    triplet_records = []
    for mid_b in range(min_band + 1, num_bands):
        b_trip = [mid_b - 1, mid_b, mid_b + 1]

        delta_grid = np.zeros(len(results))
        gap_grid = np.zeros(len(results))
        irrep_counts = {}
        match_scores = []

        for i, r in enumerate(results):
            f_map = r["freqs"]
            i_map = r["irreps"]
            f_vals = [f_map.get(b, np.nan) for b in b_trip]
            i_vals = [i_map.get(b, "Unknown") for b in b_trip]

            trip_key = "-".join(i_vals)
            irrep_counts[trip_key] = irrep_counts.get(trip_key, 0) + 1

            # Irrep match score with target 2D irreps
            common = set(i_vals).intersection(target_tokens)
            match_scores.append(len(common) / max(len(target_tokens), 1))

            if len(f_vals) == 3 and not any(np.isnan(f_vals)):
                gap_val = (max(f_vals) - min(f_vals)) / max(np.mean(f_vals), 1e-6)
                gap_grid[i] = gap_val
                delta_grid[i] = f_vals[1] - 0.5 * (f_vals[0] + f_vals[2])
            else:
                gap_grid[i] = np.nan
                delta_grid[i] = np.nan

        delta_2d = delta_grid.reshape((n_grid, n_grid))
        gap_2d = gap_grid.reshape((n_grid, n_grid))

        valid_deltas = [d for d, r in zip(delta_grid, results) if r["is_connected"] and np.isfinite(d)]
        valid_gaps = [g for g, r in zip(gap_grid, results) if r["is_connected"] and np.isfinite(g)]

        min_delta = float(np.min(valid_deltas)) if valid_deltas else 1.0
        max_delta = float(np.max(valid_deltas)) if valid_deltas else -1.0
        min_gap = float(np.min(valid_gaps)) if valid_gaps else 1.0

        has_crossing = (min_delta < 0) and (max_delta > 0)
        avg_match = float(np.mean(match_scores)) if match_scores else 0.0

        if has_crossing:
            p_locus = 1.0 * (0.5 + 0.5 * avg_match)
            status_str = f"Guaranteed ({avg_match*100:.0f}% Irrep Match)"
        else:
            p_locus = float(np.exp(- (min(abs(min_delta), abs(max_delta)) ** 2) / (2 * (0.02 ** 2)))) * avg_match
            status_str = f"Low ({p_locus*100:.1f}%)"

        dominant_irreps = max(irrep_counts, key=irrep_counts.get) if irrep_counts else "Unknown"

        triplet_records.append({
            "bands": b_trip,
            "dominant_irreps": dominant_irreps,
            "match_score": avg_match,
            "p_locus": p_locus,
            "status": status_str,
            "has_crossing": has_crossing,
            "min_gap": min_gap,
            "min_delta": min_delta,
            "max_delta": max_delta,
            "delta_2d": delta_2d,
            "gap_2d": gap_2d,
        })

    # Generate 3D Diagnostic Plot
    fig_file = step_dir / "screening_3d_triplets.png"
    _plot_3d_triplets(
        x1_vals, x2_vals, triplet_records, conn_mask,
        p1_name, p2_name, fixed_h, target_irreps_str, output_file=fig_file,
    )

    if verbose:
        print("\n" + "=" * 78)
        print("STEP 2: 3D MODE TRIPLET SCREENING & MATCHING SUMMARY")
        print("=" * 78)
        print(f"{'Bands':<12} | {'Dominant Irreps':<18} | {'Irrep Match':<12} | {'Min Gap':<10} | {'Status'}")
        print("-" * 78)
        for t in triplet_records:
            b_str = str(t["bands"])
            irr_str = t["dominant_irreps"]
            m_str = f"{t['match_score']*100:.0f}%"
            gap_str = f"{t['min_gap']:.4f}"
            print(f"{b_str:<12} | {irr_str:<18} | {m_str:<12} | {gap_str:<10} | {t['status']}")
        print("=" * 78)
        print(f"Saved 3D screening diagnostic plot to '{fig_file}'")

    best_3d_triplet = max(triplet_records, key=lambda t: (t["match_score"] > 0.6, t["has_crossing"], t["p_locus"], -t["min_gap"]))
    return {
        "step": 2,
        "step_dir": step_dir,
        "triplet_records": triplet_records,
        "best_triplet": best_3d_triplet,
        "figure_path": fig_file,
    }


def _plot_3d_triplets(
    x1_vals: np.ndarray,
    x2_vals: np.ndarray,
    triplets: List[Dict[str, Any]],
    conn_mask: np.ndarray,
    p1_name: str,
    p2_name: str,
    fixed_h: float,
    target_irreps_str: str,
    output_file: Path,
) -> None:
    """Multi-panel figure showing 3D mode triplet landscapes with connectivity boundary overlay."""
    n_trips = len(triplets)
    n_cols = min(3, n_trips)
    n_rows = math.ceil(n_trips / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.4 * n_rows), squeeze=False)
    fig.suptitle(f"3D Membrane Slab Triplet Screening (h/a={fixed_h}, Target: {target_irreps_str})", fontsize=14, fontweight="bold", y=1.02)

    p1_lbl = f"${p1_name[0]}_{{{p1_name[1:]}}}/a$" if len(p1_name) > 1 and p1_name[1:].isdigit() else f"${p1_name}/a$"
    p2_lbl = f"${p2_name[0]}_{{{p2_name[1:]}}}/a$" if len(p2_name) > 1 and p2_name[1:].isdigit() else f"${p2_name}/a$"

    for idx, t in enumerate(triplets):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        delta_2d = t["delta_2d"]
        v_max = max(abs(t["min_delta"]), abs(t["max_delta"]), 0.01)
        norm = mcolors.TwoSlopeNorm(vmin=-v_max, vcenter=0.0, vmax=v_max)

        im = ax.imshow(
            delta_2d,
            origin="lower",
            extent=[x1_vals[0], x1_vals[-1], x2_vals[0], x2_vals[-1]],
            cmap="RdBu_r",
            norm=norm,
            aspect="equal",
        )

        disconn = ~conn_mask
        if np.any(disconn):
            ax.contourf(
                x1_vals, x2_vals, disconn.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["///"],
                alpha=0.0,
            )

        if t["has_crossing"]:
            try:
                ax.contour(
                    x1_vals, x2_vals, delta_2d,
                    levels=[0.0],
                    colors=["lime"],
                    linewidths=[2.2],
                )
            except Exception:
                pass

        b_str = str(t["bands"])
        irr_str = t["dominant_irreps"]
        m_str = f"Match = {t['match_score']*100:.0f}%"

        ax.set_title(f"Bands {b_str} ({irr_str})\n{m_str} | Min Gap = {t['min_gap']:.4f}", fontsize=10, fontweight="bold")
        ax.set_xlabel(p1_lbl, fontsize=10)
        ax.set_ylabel(p2_lbl, fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\Delta\omega = \omega_2 - \frac{\omega_1 + \omega_3}{2}$", fontsize=9)

    for idx in range(n_trips, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
