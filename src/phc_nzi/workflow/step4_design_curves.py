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

    if not sweep_records:
        import json
        s3_dir = output_dir / "step3_optimization"
        sweep_records = []
        for h_dir in sorted(s3_dir.glob("h_*")):
            try:
                h_val = float(h_dir.name.replace("h_", ""))
            except ValueError:
                continue
            locus_csv = h_dir / "bo_output" / "locus_01" / "bo_locus.csv"
            best_json = h_dir / "bo_output" / "best_params.json"
            opt_p = {}
            w_d = 0.5
            if best_json.is_file():
                try:
                    with open(best_json) as f:
                        b_data = json.load(f)
                        opt_p = b_data.get("optimal_parameters", {})
                        w_d = b_data.get("freq_dirac", 0.5)
                except Exception:
                    pass
            sweep_records.append({
                "h_over_a": h_val,
                "optimal_params": opt_p,
                "optimal_freq": w_d,
                "locus_csv": locus_csv if locus_csv.is_file() else None,
            })

    summary_rows = []
    prev_r1: Optional[float] = None
    prev_r2: Optional[float] = None

    if sweep_records:
        sweep_records_sorted = sorted(sweep_records, key=lambda s: float(s["h_over_a"]))

        for s in sweep_records_sorted:
            h_a = float(s["h_over_a"])
            l_csv = s.get("locus_csv")
            r1_a = float(s["optimal_params"].get(param_names[0], 0.25))
            r2_a = float(s["optimal_params"].get(param_names[1], 0.15)) if len(param_names) > 1 else 0.15
            w_d = float(s.get("optimal_freq", 0.50))
            vg_c = 0.05

            if l_csv and Path(l_csv).is_file():
                with open(l_csv, "r") as f:
                    pts = list(csv.DictReader(f))
                if pts:
                    if prev_r1 is None:
                        # First thickness: choose point with minimum residual gap
                        best_pt = min(pts, key=lambda p: float(p.get("residual_gap", 1.0)))
                    else:
                        # Subsequent thicknesses: choose point on locus nearest to previous thickness
                        # to guarantee a continuous, smooth physical branch across the sweep
                        def dist_cost(p):
                            p_r1 = float(p.get("r1", 0.0))
                            p_r2 = float(p.get("r2", 0.0))
                            d2 = (p_r1 - prev_r1) ** 2 + (p_r2 - prev_r2) ** 2
                            gap_pen = float(p.get("residual_gap", 0.0)) * 0.01
                            return d2 + gap_pen

                        best_pt = min(pts, key=dist_cost)

                    r1_a = float(best_pt.get("r1", r1_a))
                    r2_a = float(best_pt.get("r2", r2_a))
                    vg_c = float(best_pt.get("group_velocity", 0.05))

            prev_r1 = r1_a
            prev_r2 = r2_a

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
