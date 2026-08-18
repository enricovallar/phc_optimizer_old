"""
Unit tests for Dirac-like cone wavelength scaling and universal design curves.
Validates against published analytical values in Vallar et al., OME 16, 2681 (2026).
"""

import os
import tempfile
from pathlib import Path
import numpy as np

from phc_nzi.design_curves import SlabDesignCurves, generate_design_curves


def test_scaling_invariants():
    print("--- Test 1: Mathematical Invariant Scaling ---")
    mock_data = [
        {"h_over_a": 0.27, "omega_D": 0.702, "r1_over_a": 0.220, "r2_over_a": 0.280, "vg_over_c": 0.045},
        {"h_over_a": 0.30, "omega_D": 0.684, "r1_over_a": 0.228, "r2_over_a": 0.289, "vg_over_c": 0.075},
        {"h_over_a": 0.3676, "omega_D": 0.6581, "r1_over_a": 0.2412, "r2_over_a": 0.3020, "vg_over_c": 0.120},
        {"h_over_a": 0.40, "omega_D": 0.643, "r1_over_a": 0.248, "r2_over_a": 0.311, "vg_over_c": 0.132},
        {"h_over_a": 0.45, "omega_D": 0.622, "r1_over_a": 0.258, "r2_over_a": 0.324, "vg_over_c": 0.145},
        {"h_over_a": 0.50, "omega_D": 0.601, "r1_over_a": 0.268, "r2_over_a": 0.336, "vg_over_c": 0.152},
    ]

    lambda0 = 1550.0
    curves = SlabDesignCurves(mock_data, lambda0_nm=lambda0)

    # Query for target h = 375 nm
    design_375 = curves.query(375.0)

    print(f"Target h = 375 nm @ lambda0 = {lambda0} nm:")
    print(f"  a = {design_375['a_nm']:.1f} nm (expected ~1020 nm)")
    print(f"  r1 = {design_375['r1_nm']:.1f} nm (expected ~246 nm)")
    print(f"  r2 = {design_375['r2_nm']:.1f} nm (expected ~308 nm)")
    print(f"  FF = {design_375['filling_factor']:.3f} (expected ~0.53)")
    print(f"  theta = {design_375['theta_deg']:.1f} deg (expected ~51.4 deg)")

    assert abs(design_375["a_nm"] - 1020.0) < 5.0, f"Lattice constant mismatch: {design_375['a_nm']}"
    assert abs(design_375["r1_nm"] - 246.0) < 5.0, f"r1 mismatch: {design_375['r1_nm']}"
    assert abs(design_375["r2_nm"] - 308.0) < 5.0, f"r2 mismatch: {design_375['r2_nm']}"
    print("Test 1 passed successfully!")


def test_export_and_plotting():
    print("\n--- Test 2: Dense CSV Export & Design Rules Plotting ---")
    mock_data = [
        {"h_over_a": 0.27, "omega_D": 0.702, "r1_over_a": 0.220, "r2_over_a": 0.280, "vg_over_c": 0.045},
        {"h_over_a": 0.30, "omega_D": 0.684, "r1_over_a": 0.228, "r2_over_a": 0.289, "vg_over_c": 0.075},
        {"h_over_a": 0.3676, "omega_D": 0.6581, "r1_over_a": 0.2412, "r2_over_a": 0.3020, "vg_over_c": 0.120},
        {"h_over_a": 0.40, "omega_D": 0.643, "r1_over_a": 0.248, "r2_over_a": 0.311, "vg_over_c": 0.132},
        {"h_over_a": 0.45, "omega_D": 0.622, "r1_over_a": 0.258, "r2_over_a": 0.324, "vg_over_c": 0.145},
        {"h_over_a": 0.50, "omega_D": 0.601, "r1_over_a": 0.268, "r2_over_a": 0.336, "vg_over_c": 0.152},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_csv = Path(tmpdir) / "slab_sweep_summary.csv"
        curves = SlabDesignCurves(mock_data, lambda0_nm=1550.0)

        # Export table
        dense_csv = Path(tmpdir) / "design_rules_1550nm.csv"
        curves.export_dense_csv(dense_csv, n_points=50)
        assert dense_csv.is_file(), "Dense CSV not created"

        # Generate figure
        fig_path = Path(tmpdir) / "design_rules_1550nm.png"
        curves.plot_design_rules(output_path=fig_path, highlight_h_nm=375.0)
        assert fig_path.is_file(), "Design rules figure not created"
        assert fig_path.stat().st_size > 10000, "Figure file unexpectedly small"

        print(f"Generated design rules plot ({fig_path.stat().st_size} bytes)")
    print("Test 2 passed successfully!")


if __name__ == "__main__":
    test_scaling_invariants()
    test_export_and_plotting()
    print("\nALL DESIGN CURVE TESTS PASSED!")
