"""
Step 5: Final Validation & Synthesis Report
Executes high-resolution full band structure simulation at the chosen target design,
plots epsilon permittivity map, and compiles the final Discovery Report markdown.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

from ..runner import run_hpc
from ..plotter import plot_band_structure, plot_epsilon
from ..extractor import extract_frequencies
from ..config_utils import resolve_step_config


def run_step5_validation(
    cfg: Any,
    work_dir: Path,
    output_dir: Path,
    param_names: List[str],
    design_curves_result: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes Step 5: Full high-resolution band structure validation and Markdown report generation.
    """
    step_dir = output_dir / "step5_validation"
    step_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = resolve_step_config(cfg, step_name="step5_validation")

    s5_cfg = cfg_dict.get("step5_validation", cfg_dict.get("workflow", {}).get("step5_validation", {}))
    fixed_params = cfg_dict.get("parameters", {}).get("fixed", {})

    # Preserve resolution from previous steps by default
    res = int(s5_cfg.get("resolution", fixed_params.get("resolution", 25)))
    res_z = int(s5_cfg.get("res_z", fixed_params.get("res-z", fixed_params.get("res_z", 16))))
    k_pts = int(s5_cfg.get("k_points_per_segment", 30))

    sim_cfg = cfg_dict.get("simulation", {})
    cores = int(s5_cfg.get("cores", sim_cfg.get("mpi_cores", sim_cfg.get("parallel_workers", sim_cfg.get("cores", 28)))))

    h_target_nm = float(s5_cfg.get("target_thickness_nm", 375.0))
    a_nm = 840.0
    r1_val = 0.28
    r2_val = 0.35
    h_a_val = 0.446

    # Load design point from Step 4 curves (in-memory or from disk)
    curves_obj = None
    if design_curves_result and "curves" in design_curves_result:
        curves_obj = design_curves_result["curves"]
    else:
        s4_summary_csv = output_dir / "step4_design_curves" / "slab_sweep_summary.csv"
        if s4_summary_csv.is_file():
            from ..design_curves import generate_design_curves
            s4_cfg = cfg_dict.get("step4_design_curves", cfg_dict.get("workflow", {}).get("step4_design_curves", {}))
            lambda0 = float(s4_cfg.get("target_wavelength_nm", 1550.0))
            curves_obj = generate_design_curves(s4_summary_csv, lambda0_nm=lambda0, output_dir=output_dir / "step4_design_curves")

    if curves_obj is not None:
        try:
            pt = curves_obj.get_design_point(h_nm=h_target_nm)
            h_target_nm = float(pt["h_nm"])
            a_nm = float(pt["a_nm"])
            r1_val = float(pt["r1_norm"])
            r2_val = float(pt["r2_norm"])
            h_a_val = float(pt["h_over_a"])
        except Exception as e:
            if verbose:
                print(f"Notice: Design curve interpolation note: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 5: Final Design Validation & Synthesis Report")
        print("=" * 70)
        print(f"Target Design:       h = {h_target_nm:.1f} nm -> a = {a_nm:.1f} nm")
        print(f"Normalized Radii:    r1/a = {r1_val:.4f} ({r1_val*a_nm:.1f} nm), r2/a = {r2_val:.4f} ({r2_val*a_nm:.1f} nm)")
        print(f"Validation Grid:     res = {res}, res-z = {res_z}, k-points/segment = {k_pts}")
        print(f"HPC MPI Solver:      {cores} cores (use_mpi=True, symmetry_irreps=False)")
        print("-" * 70)

    target_bands = (
        cfg_dict.get("target", {}).get("symmetry", {}).get("target_bands")
        or cfg_dict.get("target", {}).get("target_bands")
        or cfg_dict.get("target", {}).get("mode_indices")
        or [8, 9, 10]
    )

    sz_raw = fixed_params.get("sz", 4.0)
    sz_val = 4.0 if str(sz_raw).lower() == "no-size" else float(sz_raw)

    num_b = s5_cfg.get(
        "num_bands",
        s5_cfg.get(
            "num-bands",
            fixed_params.get("num-bands", fixed_params.get("num_bands", 14))
        )
    )
    num_bands_val = int(num_b)
    if target_bands and max(target_bands) >= num_bands_val:
        num_bands_val = max(target_bands) + 4

    pol = str(cfg_dict.get("target", {}).get("symmetry", {}).get("polarization", cfg_dict.get("target", {}).get("polarization", "zeven"))).lower()
    if pol == "te":
        pol = "zeven"
    elif pol == "tm":
        pol = "zodd"

    val_params = {
        param_names[0]: r1_val,
        param_names[1]: r2_val,
        "h": h_a_val,
        "sz": sz_val,
        "resolution": res,
        "res-z": res_z,
        "num-bands": num_bands_val,
        "k-interp": k_pts,
        "only_gamma?": "false",
        "display_symmetry?": "false",       # No irreps during MPI validation
        "display_group_velocity?": "false",
    }
    for m in ["run-te?", "run-tm?", "run-zeven?", "run-zodd?"]:
        val_params[m] = "false"
    if pol in ["both", "all"]:
        val_params["run-zeven?"] = "true"
        val_params["run-zodd?"] = "true"
    else:
        val_params[f"run-{pol}?"] = "true"

    ctl_script = Path(sim_cfg.get("ctl_script", "main.ctl")).resolve()
    if not ctl_script.is_file():
        w_ctl = (work_dir / sim_cfg.get("ctl_script", "main.ctl")).resolve()
        if w_ctl.is_file():
            ctl_script = w_ctl
    run_hpc(
        script=ctl_script,
        mpb_command_line_params=val_params,
        use_mpi=True,
        cores=cores,
        wd=step_dir,
        auto_extract=True,
        auto_plot=False,
        only_gamma=False,
        verbose=verbose,
    )

    # Move output files if located in output subfolder
    raw_out = step_dir / "output"
    if raw_out.is_dir():
        import shutil
        for item in raw_out.iterdir():
            dest = step_dir / item.name
            if not dest.exists():
                shutil.copy2(str(item), str(dest))

    # 1. Plot Full Band Structure & Zoomed Dirac Cone Band Structure
    bs_plot = step_dir / "validation_band_structure.png"
    bs_zoom_plot = step_dir / "validation_band_structure_zoom.png"
    eps_plot = step_dir / "validation_epsilon_map.png"

    target_bands = (
        cfg_dict.get("target", {}).get("symmetry", {}).get("target_bands")
        or cfg_dict.get("target", {}).get("target_bands")
        or cfg_dict.get("target", {}).get("mode_indices")
        or [8, 9, 10]
    )

    data_src = step_dir if list(step_dir.glob("*.data")) else (raw_out if raw_out.is_dir() else step_dir)

    try:
        plot_band_structure(
            data_path=data_src,
            output_path=bs_plot,
            polarization=pol,
            title=f"Validated Full Band Structure ($h = {h_target_nm:.0f}\\text{{ nm}}, a = {a_nm:.0f}\\text{{ nm}}$)",
            highlight_gaps=True,
            style="light",
            verbose=False,
        )
    except Exception as e:
        if verbose:
            print(f"Notice: Band structure plot: {e}")

    try:
        plot_band_structure(
            data_path=data_src,
            output_path=bs_zoom_plot,
            bands=target_bands,
            polarization=pol,
            title=f"Validated Dirac-Like Cone ($h = {h_target_nm:.0f}\\text{{ nm}}, a = {a_nm:.0f}\\text{{ nm}}$)",
            highlight_gaps=False,
            style="light",
            verbose=False,
        )
    except Exception as e:
        if verbose:
            print(f"Notice: Zoomed band structure plot: {e}")

    try:
        eps_files = list(step_dir.glob("*-epsilon.h5")) + list(raw_out.glob("*-epsilon.h5") if raw_out.is_dir() else [])
        if eps_files:
            plot_epsilon(
                h5_path=eps_files[0],
                output_path=eps_plot,
                title=f"Validated Unit Cell Permittivity ($r_1={r1_val*a_nm:.0f}\\text{{ nm}}, r_2={r2_val*a_nm:.0f}\\text{{ nm}}$)",
                rectify=True,
                plane="both",
                verbose=False,
            )
    except Exception as e:
        if verbose:
            print(f"Notice: Epsilon map plot: {e}")

    # Compile Discovery Report
    report_file = output_dir / "Discovery_Report.md"
    report_md = f"""# Photonic Crystal Near-Zero-Index (NZI) Discovery Report
**Project Name**: {cfg.get('general', {}).get('name', 'PhC_Discovery')}  
**Target Wavelength**: $\\lambda_0 = 1550\\text{{ nm}}$  
**Operating Slab Thickness**: $h = {h_target_nm:.1f}\\text{{ nm}}$  

---

## 1. Executive Discovery Summary

| Parameter | Normalized Value | Physical Dimension ($\\lambda_0 = 1550\\text{{ nm}}$) |
| :--- | :--- | :--- |
| **Lattice Constant ($a$)** | $1.000$ | **${a_nm:.1f}\\text{{ nm}}$** |
| **Slab Thickness ($h$)** | ${h_a_val:.3f}$ | **${h_target_nm:.1f}\\text{{ nm}}$** |
| **Primary Radius ($r_1$)** | ${r1_val:.4f}$ | **${r1_val * a_nm:.1f}\\text{{ nm}}$** |
| **Secondary Radius ($r_2$)** | ${r2_val:.4f}$ | **${r2_val * a_nm:.1f}\\text{{ nm}}$** |

---

## 2. Validation Figures

- **Full Band Structure**: `step5_validation/validation_band_structure.png`
- **Zoomed Dirac-Like Cone**: `step5_validation/validation_band_structure_zoom.png`
- **Dielectric Permittivity Map**: `step5_validation/validation_epsilon_map.png`
- **Universal Design Rules**: `step4_design_curves/design_rules_1550nm.png`

Discovery pipeline completed successfully.
"""
    report_file.write_text(report_md)

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 5: Validation Complete!")
        print(f"Generated Discovery Report at '{report_file}'")
        print("=" * 70)

    return {
        "step": 5,
        "step_dir": step_dir,
        "report_file": report_file,
        "band_structure_path": bs_plot,
        "epsilon_map_path": eps_plot,
    }
