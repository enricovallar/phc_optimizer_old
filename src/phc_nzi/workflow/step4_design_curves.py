"""
Step 4: Universal Design Curves & Physical Scaling
Builds continuous scaled design curves for manufacturing at target wavelength (e.g. 1550 nm).
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..design_curves import generate_design_curves, SlabDesignCurves


def run_step4_design_curves(
    cfg: Any,
    work_dir: Path,
    output_dir: Path,
    param_names: List[str],
    sweep_records: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes Step 4: Universal Design Curve Generation at target wavelength lambda0.
    """
    step_dir = output_dir / "step4_design_curves"
    step_dir.mkdir(parents=True, exist_ok=True)

    s4_cfg = cfg.get("step4_design_curves", cfg.get("workflow", {}).get("step4_design_curves", {}))
    lambda0 = float(s4_cfg.get("target_wavelength_nm", 1550.0))
    highlight_h = float(s4_cfg.get("highlight_h_nm", 375.0))

    summary_rows = []
    if sweep_records:
        for s in sweep_records:
            h_a = s["h_over_a"]
            l_csv = s.get("locus_csv")
            r1_a = s["optimal_params"].get(param_names[0], 0.25)
            r2_a = s["optimal_params"].get(param_names[1], 0.15) if len(param_names) > 1 else 0.15
            w_d = s.get("optimal_freq", 0.50)
            vg_c = 0.05

            if l_csv and Path(l_csv).is_file():
                with open(l_csv, "r") as f:
                    reader = list(csv.DictReader(f))
                    if reader:
                        best_pt = min(reader, key=lambda p: float(p.get("residual_gap", 1.0)))
                        r1_a = float(best_pt.get("r1", r1_a))
                        r2_a = float(best_pt.get("r2", r2_a))
                        vg_c = float(best_pt.get("group_velocity", 0.05))

            summary_rows.append({
                "h_over_a": h_a,
                "r1_over_a": r1_a,
                "r2_over_a": r2_a,
                "omega_D": w_d,
                "vg_over_c": vg_c,
            })

    sum_csv_file = step_dir / "slab_sweep_summary.csv"
    with open(sum_csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["h_over_a", "r1_over_a", "r2_over_a", "omega_D", "vg_over_c"])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    curves = generate_design_curves(
        sweep_summary_path=sum_csv_file,
        lambda0_nm=lambda0,
        output_dir=step_dir,
        highlight_h_nm=highlight_h,
    )

    fig_path = step_dir / f"design_rules_{int(lambda0)}nm.png"
    csv_path = step_dir / f"design_rules_{int(lambda0)}nm.csv"

    if verbose:
        print("\n" + "=" * 70)
        print(f"STEP 4: Universal Design Curves (Target λ0 = {lambda0:.1f} nm)")
        print("=" * 70)
        print(f"Saved design rules figure to '{fig_path}'")
        print(f"Saved dense CSV lookup table to '{csv_path}'")
        print("=" * 70)

    return {
        "step": 4,
        "step_dir": step_dir,
        "curves": curves,
        "summary_csv": sum_csv_file,
        "figure_path": fig_path,
        "csv_path": csv_path,
    }
