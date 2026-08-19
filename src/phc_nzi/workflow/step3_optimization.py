"""
Step 3: Multi-Thickness Bayesian Optimization Sweep
Executes Bayesian Optimization sweeps across slab thicknesses h/a in [0.30..0.50].
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from ..bo import BayesianOptimizer


def run_step3_optimization(
    cfg: Any,
    work_dir: Path,
    output_dir: Path,
    param_names: List[str],
    param_bounds: List[List[float]],
    fixed_params: Dict[str, Any],
    sim_cfg: Dict[str, Any],
    target_irreps: Optional[List[str]] = None,
    irrep_occurrences: Optional[List[int]] = None,
    target_modes: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes Step 3: Multi-Thickness Bayesian Optimization Sweep across h/a
    using dynamic target irreps and occurrence matching discovered in Step 2.
    """
    step_dir = output_dir / "step3_optimization"
    step_dir.mkdir(parents=True, exist_ok=True)

    s3_cfg = cfg.get("step3_optimization", cfg.get("workflow", {}).get("step3_optimization", {}))
    h_sweep = list(s3_cfg.get("h_sweep", [0.30, 0.35, 0.40, 0.45, 0.50]))

    eff_irreps = target_irreps or list(cfg.target.symmetry.get("target_irreps", ["A_2", "E", "E"]))
    eff_occs = irrep_occurrences or list(cfg.target.symmetry.get("irrep_occurrences", [2, 3, 3]))

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: Multi-Thickness Bayesian Optimization Sweep")
        print("=" * 70)
        print(f"Target Irreps:       {eff_irreps}")
        print(f"Irrep Occurrences:   {eff_occs}")
        print(f"Thickness Sweep:     h/a in {h_sweep}")
        print(f"Workers:             {sim_cfg.get('parallel_workers', 28)} concurrent workers")
        print("-" * 70)

    sweep_records = []
    for h_val in h_sweep:
        h_dir = step_dir / f"h_{h_val:.2f}"
        h_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n---> Launching Bayesian Optimization for h/a = {h_val:.2f} in '{h_dir.name}'...")

        cfg_h = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_h.general.output_dir = str(h_dir / "bo_output")
        cfg_h.simulation.work_dir = str(h_dir)
        cfg_h.parameters.fixed.h = float(h_val)
        sz_raw = fixed_params.get("sz", 4.0)
        sz_val = 4.0 if str(sz_raw).lower() == "no-size" else float(sz_raw)
        cfg_h.parameters.fixed.sz = sz_val
        cfg_h.parameters.fixed["num-bands"] = int(s3_cfg.get("num_bands", fixed_params.get("num-bands", fixed_params.get("num_bands", 14))))
        cfg_h.parameters.fixed["resolution"] = int(fixed_params.get("resolution", 25))
        cfg_h.parameters.fixed["res-z"] = int(fixed_params.get("res-z", fixed_params.get("res_z", 16)))

        # Configure dynamic irrep and occurrence tracking (no hardcoded band indices)
        cfg_h.target.symmetry.target_irreps = list(eff_irreps)
        cfg_h.target.symmetry.irrep_occurrences = list(eff_occs)
        cfg_h.target.symmetry.min_band = 2
        cfg_h.target.symmetry.bypass_irrep_identification = False
        cfg_h.target.symmetry.bypass_symmetry = False
        cfg_h.target.symmetry.mode_indices = None

        cfg_h.postprocessing.enabled = True
        cfg_h.postprocessing.group_velocity.enabled = True

        cfg_h_dict = OmegaConf.to_container(cfg_h, resolve=True)
        bo_inst = BayesianOptimizer(cfg_h_dict)
        bo_res = bo_inst.run()

        locus_files = list(Path(cfg_h.general.output_dir).glob("locus_*/bo_locus.csv"))
        locus_csv = locus_files[0] if locus_files else None
        sweep_records.append({
            "h_over_a": float(h_val),
            "bo_dir": h_dir,
            "locus_csv": locus_csv if locus_csv.is_file() else None,
            "optimal_params": bo_res.get("optimal_parameters", {}),
            "optimal_cost": bo_res.get("raw_cost"),
            "optimal_freq": bo_res.get("freq_dirac"),
        })

    # Summary Plot
    fig_file = step_dir / "bo_sweep_summary.png"
    _plot_sweep_summary(sweep_records, param_names, output_file=fig_file)

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: Multi-Thickness Optimization Sweep Complete!")
        print(f"Saved summary figure to '{fig_file}'")
        print("=" * 70)

    return {
        "step": 3,
        "step_dir": step_dir,
        "sweep_records": sweep_records,
        "figure_path": fig_file,
    }


def _plot_sweep_summary(sweep_records: List[Dict[str, Any]], param_names: List[str], output_file: Path) -> None:
    """Plots multi-thickness overlay of extracted loci and group velocities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    p1_name = param_names[0]
    p2_name = param_names[1] if len(param_names) > 1 else "r2"
    p1_lbl = f"${p1_name[0]}_{{{p1_name[1:]}}}/a$" if len(p1_name) > 1 and p1_name[1:].isdigit() else f"${p1_name}/a$"
    p2_lbl = f"${p2_name[0]}_{{{p2_name[1:]}}}/a$" if len(p2_name) > 1 and p2_name[1:].isdigit() else f"${p2_name}/a$"

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(sweep_records), 1)))

    for idx, s in enumerate(sweep_records):
        h_val = s["h_over_a"]
        c = colors[idx]
        l_csv = s.get("locus_csv")

        if l_csv and Path(l_csv).is_file():
            with open(l_csv, "r") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    r1 = [float(r["r1"]) for r in rows]
                    r2 = [float(r["r2"]) for r in rows]
                    vg = [float(r.get("group_velocity", 0.0)) for r in rows]
                    t_norm = np.linspace(0, 1, len(r1))

                    ax1.plot(r1, r2, color=c, linewidth=2.0, label=f"$h/a = {h_val:.2f}$")
                    ax1.scatter(r1[0], r2[0], color=c, marker="o", s=30)
                    ax2.plot(t_norm, vg, color=c, linewidth=2.0, label=f"$h/a = {h_val:.2f}$")

    ax1.set_title("(a) Degeneracy Loci vs Slab Thickness", fontsize=12, fontweight="bold")
    ax1.set_xlabel(p1_lbl, fontsize=11)
    ax1.set_ylabel(p2_lbl, fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.4, linestyle="--")

    ax2.set_title("(b) Group Velocity Tuning $v_g(t)$ vs $h/a$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Normalized Arc Trajectory $t \\in [0, 1]$", fontsize=11)
    ax2.set_ylabel("Group Velocity $v_g / c$", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.4, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
