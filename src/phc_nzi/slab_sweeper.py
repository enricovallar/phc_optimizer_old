"""
Multi-Thickness Slab Sweeper & Master Degeneracy Database Engine
Executes Bayesian Optimization sweeps across normalized slab thicknesses h/a
and aggregates optimal Dirac-like cone loci for wavelength-tuning design curves.
"""

import os
import copy
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from phc_nzi.bo import BayesianOptimizer, load_bo_config
from phc_nzi.design_curves import generate_design_curves, SlabDesignCurves


def run_slab_sweep(
    config_path: Union[str, os.PathLike],
    h_ratios: Optional[List[float]] = None,
    work_dir: Optional[Union[str, os.PathLike]] = None,
    output_dir_name: str = "sweep_results",
    target_wavelength_nm: float = 1550.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Sweeps Bayesian Optimization across multiple normalized slab thickness ratios h/a.

    Parameters:
    -----------
    config_path : str or Path
        Path to base bo_config.yaml.
    h_ratios : list of float, optional
        List of normalized slab thicknesses h/a (default: [0.30, 0.35, 0.40, 0.45, 0.50]).
    work_dir : str or Path, optional
        Base root directory for the sweep (defaults to parent directory of config file).
    output_dir_name : str, default "sweep_results"
        Subfolder name to store all sweep runs.
    target_wavelength_nm : float, default 1550.0
        Target operational wavelength in nanometers.
    verbose : bool, default True
        Whether to print progress and status logs.

    Returns:
    --------
    dict
        Summary dictionary containing list of optimal points and path to design curves.
    """
    if h_ratios is None:
        h_ratios = [0.30, 0.35, 0.40, 0.45, 0.50]

    base_config = load_bo_config(config_path)

    cfg_dir = Path(config_path).resolve().parent
    root_work = Path(work_dir).resolve() if work_dir else cfg_dir
    sweep_root = root_work / output_dir_name
    sweep_root.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, Any]] = []

    if verbose:
        print("=" * 78)
        print(f"Starting 3D Slab Thickness Sweep across {len(h_ratios)} ratios: {h_ratios}")
        print(f"Sweep root directory: '{sweep_root}'")
        print("=" * 78)

    ctl_script_name = base_config["simulation"].get("ctl_script", "main.ctl")
    source_ctl = cfg_dir / ctl_script_name
    if not source_ctl.is_file():
        source_ctl = root_work / ctl_script_name

    for idx, h_val in enumerate(h_ratios):
        h_a = float(h_val)
        run_name = f"h_{h_a:.2f}"
        run_dir = sweep_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Ensure main.ctl is present in run_dir
        target_ctl = run_dir / ctl_script_name
        if source_ctl.is_file() and not target_ctl.is_file():
            target_ctl.write_text(source_ctl.read_text())

        # Configure sub-run dictionary
        sub_cfg = copy.deepcopy(base_config)
        sub_cfg["simulation"]["work_dir"] = str(run_dir)
        sub_cfg["simulation"]["output_dir"] = "bo_output"
        sub_cfg["simulation"]["ctl_script"] = ctl_script_name
        sub_cfg["fixed_parameters"]["h"] = h_a
        sub_cfg["fixed_parameters"]["sz"] = sub_cfg["fixed_parameters"].get("sz", 4)
        sub_cfg["fixed_parameters"]["num-bands"] = sub_cfg["fixed_parameters"].get("num-bands", 12)

        if verbose:
            print(f"\n[{idx + 1}/{len(h_ratios)}] Running optimization for h/a = {h_a:.2f} in '{run_dir}'...")

        opt = BayesianOptimizer(sub_cfg)
        bo_result = opt.run()

        # Extract optimal Dirac cone point from locus or optimization records
        opt_point = _extract_optimal_sweep_point(opt, h_a)
        summary_records.append(opt_point)

        if verbose:
            print(f"[{idx + 1}/{len(h_ratios)}] h/a = {h_a:.2f} completed! Optimal point: "
                  f"r1/a={opt_point['r1_over_a']:.4f}, r2/a={opt_point['r2_over_a']:.4f}, "
                  f"omega_D={opt_point['omega_D']:.4f}, vg/c={opt_point['vg_over_c']:.4f}, FF={opt_point['filling_factor']:.4f}")

    # Export master sweep summary CSV and JSON
    summary_csv = sweep_root / "slab_sweep_summary.csv"
    summary_json = sweep_root / "slab_sweep_summary.json"

    _export_sweep_summary(summary_records, summary_csv, summary_json)

    if verbose:
        print("\n" + "=" * 78)
        print(f"Master sweep summary saved to '{summary_csv}'")
        print("Generating physical design curves for lambda_0 = {target_wavelength_nm} nm...")

    # Generate design rules figure and continuous curve interpolators
    curves = generate_design_curves(
        sweep_summary_path=summary_csv,
        lambda0_nm=target_wavelength_nm,
        output_dir=sweep_root,
        highlight_h_nm=375.0,
    )

    return {
        "sweep_root": sweep_root,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "points": summary_records,
        "curves": curves,
    }


def _extract_optimal_sweep_point(opt: BayesianOptimizer, h_a: float) -> Dict[str, Any]:
    """
    Extracts the highest-quality Dirac-like cone point from the extracted locus
    or the lowest-cost evaluated point from the optimization history.
    """
    locus_csv = opt.output_dir / "bo_locus.csv"
    best_record = None

    if locus_csv.is_file():
        locus_points = []
        with open(locus_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {}
                for k, v in row.items():
                    k_c = k.strip().lower().replace(" ", "_")
                    try:
                        cleaned[k_c] = float(v.strip())
                    except ValueError:
                        cleaned[k_c] = v.strip()
                locus_points.append(cleaned)

        if locus_points:
            # Filter for valid points (or points passing connectivity)
            valid_points = [p for p in locus_points if str(p.get("valid", "True")).lower() == "true"]
            candidates = valid_points if valid_points else locus_points

            # Preference: Point with highest group velocity vg
            vg_candidates = [p for p in candidates if "vg" in p and not np.isnan(p["vg"])]
            if vg_candidates:
                best_record = max(vg_candidates, key=lambda p: float(p["vg"]))
            else:
                # Minimum residual gap / highest FOM
                best_record = max(candidates, key=lambda p: float(p.get("fom", 0.0)))

    if best_record is not None:
        r1_a = float(best_record.get("r1", 0.25))
        r2_a = float(best_record.get("r2", 0.15))
        vg = float(best_record.get("vg", 0.10)) if not np.isnan(float(best_record.get("vg", np.nan))) else 0.10
        # Determine omega_D at Gamma for this point
        gap, cost, bands, full_map = opt._evaluate_point_gamma_gap({"r1": r1_a, "r2": r2_a})
        omega_D = float(np.mean([full_map[b][2] for b in bands if b in full_map])) if full_map else 0.50
    else:
        # Fallback to best record in opt.records
        valid_records = [r for r in opt.records if r.get("raw_cost") is not None and float(r["raw_cost"]) < 1.0]
        if not valid_records:
            valid_records = opt.records

        best_opt = min(valid_records, key=lambda r: float(r.get("raw_cost", 1.0)))
        r1_a = float(best_opt["params"].get("r1", 0.25))
        r2_a = float(best_opt["params"].get("r2", 0.15))
        omega_D = float(best_opt.get("freq_middle", 0.50))
        vg = float(best_opt.get("vg", 0.10)) if best_opt.get("vg") is not None else 0.10

    ff = float(1.0 - np.pi * (r1_a**2 + r2_a**2))

    return {
        "h_over_a": h_a,
        "omega_D": omega_D,
        "r1_over_a": r1_a,
        "r2_over_a": r2_a,
        "vg_over_c": vg,
        "filling_factor": ff,
    }


def _export_sweep_summary(
    records: List[Dict[str, Any]],
    csv_path: Path,
    json_path: Path
) -> None:
    """Exports master sweep summary table to CSV and JSON."""
    fieldnames = ["h_over_a", "omega_D", "r1_over_a", "r2_over_a", "vg_over_c", "filling_factor"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "h_over_a": f"{r['h_over_a']:.4f}",
                "omega_D": f"{r['omega_D']:.6f}",
                "r1_over_a": f"{r['r1_over_a']:.6f}",
                "r2_over_a": f"{r['r2_over_a']:.6f}",
                "vg_over_c": f"{r['vg_over_c']:.6f}",
                "filling_factor": f"{r['filling_factor']:.6f}",
            })

    with open(json_path, "w") as f:
        json.dump({"num_points": len(records), "points": records}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run 3D Slab Thickness Sweeps for Dirac-like Cone Wavelength Tuning."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="InP slab/bo_config.yaml",
        help="Path to base bo_config.yaml file"
    )
    parser.add_argument(
        "--h-ratios",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50],
        help="List of normalized slab thickness ratios h/a to evaluate (default: 0.30 0.35 0.40 0.45 0.50)"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="Base working directory for execution"
    )
    parser.add_argument(
        "--wavelength", "-w",
        type=float,
        default=1550.0,
        help="Target wavelength lambda_0 in nanometers (default: 1550.0 nm)"
    )

    args = parser.parse_args()
    run_slab_sweep(
        config_path=args.config,
        h_ratios=args.h_ratios,
        work_dir=args.dir,
        target_wavelength_nm=args.wavelength,
    )


if __name__ == "__main__":
    main()
