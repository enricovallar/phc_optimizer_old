"""
Dirac-like Cone Wavelength Scaling and Universal Design Curves Engine
Based on Vallar et al., Optical Materials Express 16, 2681-2695 (2026).
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


class SlabDesignCurves:
    """
    Manages physical scaling invariants and continuous interpolation of design
    curves for 3D photonic crystal slabs at target operational wavelengths.
    """

    def __init__(
        self,
        sweep_data: Union[str, os.PathLike, List[Dict[str, Any]], Dict[str, Any]],
        lambda0_nm: float = 1550.0,
    ):
        """
        Initialize design curves from sweep summary data.

        Parameters:
        -----------
        sweep_data : str, Path, or list of dicts
            Path to slab_sweep_summary.csv / .json, or a list of record dicts.
            Each record must contain:
              - 'h_over_a': normalized slab thickness h/a
              - 'omega_D' (or 'freq_middle'): normalized Dirac frequency omega*a/(2pi*c)
              - 'r1_over_a' (or 'r1'): normalized radius r1/a
              - 'r2_over_a' (or 'r2'): normalized radius r2/a
              - 'vg_over_c' (or 'vg'): group velocity vg/c (optional)
              - 'filling_factor' (or 'FF'): dielectric filling factor (optional)
        lambda0_nm : float
            Target operating wavelength in nanometers (default: 1550.0 nm).
        """
        self.lambda0_nm = float(lambda0_nm)
        self.raw_records = self._load_data(sweep_data)
        if len(self.raw_records) < 2:
            raise ValueError(f"Need at least 2 distinct h/a points to construct design curves, got {len(self.raw_records)}.")

        # Sort records by h_over_a
        self.raw_records.sort(key=lambda r: float(r["h_over_a"]))

        # Build physical scaled arrays
        self._build_scaled_points()
        self._fit_interpolators()

    def _load_data(self, data_source: Any) -> List[Dict[str, Any]]:
        """Parse data from CSV, JSON, or Python list."""
        if isinstance(data_source, (str, Path, os.PathLike)):
            path = Path(data_source).resolve()
            if not path.is_file():
                # If directory provided, look for standard summary files
                if path.is_dir():
                    csv_path = path / "slab_sweep_summary.csv"
                    json_path = path / "slab_sweep_summary.json"
                    if csv_path.is_file():
                        path = csv_path
                    elif json_path.is_file():
                        path = json_path
                    else:
                        raise FileNotFoundError(f"No sweep summary found in '{path}'")
                else:
                    raise FileNotFoundError(f"Sweep summary file not found: '{path}'")

            if path.suffix == ".json":
                with open(path, "r") as f:
                    content = json.load(f)
                    return content.get("points", content) if isinstance(content, dict) else content
            else:
                # Parse CSV
                records = []
                with open(path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cleaned = {}
                        for k, v in row.items():
                            k_clean = k.strip().lower().replace(" ", "_")
                            try:
                                cleaned[k_clean] = float(v.strip())
                            except ValueError:
                                cleaned[k_clean] = v.strip()
                        records.append(cleaned)
                return records

        elif isinstance(data_source, dict):
            raw_list = data_source.get("points", [data_source])
        elif isinstance(data_source, list):
            raw_list = list(data_source)
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")

        records = []
        for item in raw_list:
            if isinstance(item, dict):
                cleaned = {str(k).strip().lower().replace(" ", "_"): v for k, v in item.items()}
                records.append(cleaned)
        return records

    def _build_scaled_points(self) -> None:
        """Apply the universal wavelength scaling laws from Vallar et al. (OME 2026)."""
        h_over_a_list = []
        omega_D_list = []
        r1_over_a_list = []
        r2_over_a_list = []
        vg_list = []
        ff_list = []

        a_nm_list = []
        h_nm_list = []
        r1_nm_list = []
        r2_nm_list = []
        rho_nm_list = []
        theta_deg_list = []

        for r in self.raw_records:
            h_a = float(r.get("h_over_a", r.get("h/a", r.get("h", 0.35))))
            omega = float(r.get("omega_d", r.get("omega_n", r.get("freq_middle", r.get("frequency", r.get("freq", 0.5))))))
            r1_a = float(r.get("r1_over_a", r.get("r1/a", r.get("r1", 0.25))))
            r2_a = float(r.get("r2_over_a", r.get("r2/a", r.get("r2", 0.15))))
            vg = float(r.get("vg_over_c", r.get("vg/c", r.get("group_velocity", r.get("vg", 0.1)))))

            # Filling factor
            if "filling_factor" in r or "ff" in r:
                ff = float(r.get("filling_factor", r.get("ff")))
            else:
                ff = float(1.0 - np.pi * (r1_a**2 + r2_a**2))

            # Physical scaling laws:
            # 1. a = omega_D * lambda_0
            a_nm = omega * self.lambda0_nm
            # 2. h = (h/a) * a
            h_nm = h_a * a_nm
            # 3. r1 = (r1/a) * a, r2 = (r2/a) * a
            r1_nm = r1_a * a_nm
            r2_nm = r2_a * a_nm

            # Polar coordinates
            rho_norm = np.sqrt(r1_a**2 + r2_a**2)
            rho_nm = rho_norm * a_nm
            theta_deg = np.degrees(np.arctan2(r2_a, r1_a))

            h_over_a_list.append(h_a)
            omega_D_list.append(omega)
            r1_over_a_list.append(r1_a)
            r2_over_a_list.append(r2_a)
            vg_list.append(vg)
            ff_list.append(ff)

            a_nm_list.append(a_nm)
            h_nm_list.append(h_nm)
            r1_nm_list.append(r1_nm)
            r2_nm_list.append(r2_nm)
            rho_nm_list.append(rho_nm)
            theta_deg_list.append(theta_deg)

        self.h_over_a = np.array(h_over_a_list)
        self.omega_D = np.array(omega_D_list)
        self.r1_over_a = np.array(r1_over_a_list)
        self.r2_over_a = np.array(r2_over_a_list)
        self.vg = np.array(vg_list)
        self.ff = np.array(ff_list)

        self.a_nm = np.array(a_nm_list)
        self.h_nm = np.array(h_nm_list)
        self.r1_nm = np.array(r1_nm_list)
        self.r2_nm = np.array(r2_nm_list)
        self.rho_nm = np.array(rho_nm_list)
        self.theta_deg = np.array(theta_deg_list)

    def _fit_interpolators(self) -> None:
        """Fit shape-preserving PCHIP spline interpolators as functions of slab thickness h (nm)."""
        # Ensure strictly monotonic h_nm for interpolation
        sort_idx = np.argsort(self.h_nm)
        h_sorted = self.h_nm[sort_idx]

        self.h_min = float(h_sorted[0])
        self.h_max = float(h_sorted[-1])

        self.interp_a = PchipInterpolator(h_sorted, self.a_nm[sort_idx])
        self.interp_r1 = PchipInterpolator(h_sorted, self.r1_nm[sort_idx])
        self.interp_r2 = PchipInterpolator(h_sorted, self.r2_nm[sort_idx])
        self.interp_rho = PchipInterpolator(h_sorted, self.rho_nm[sort_idx])
        self.interp_theta = PchipInterpolator(h_sorted, self.theta_deg[sort_idx])
        self.interp_ff = PchipInterpolator(h_sorted, self.ff[sort_idx])
        self.interp_vg = PchipInterpolator(h_sorted, self.vg[sort_idx])
        self.interp_h_over_a = PchipInterpolator(h_sorted, self.h_over_a[sort_idx])
        self.interp_omega = PchipInterpolator(h_sorted, self.omega_D[sort_idx])

    def query(self, h_nm: float) -> Dict[str, float]:
        """
        Query the design curve for a specific physical slab thickness h (in nm).

        Parameters:
        -----------
        h_nm : float
            Physical slab thickness in nanometers (e.g. 375 nm).

        Returns:
        --------
        dict
            Exact design parameters: 'a', 'r1', 'r2', 'h', 'FF', 'rho_nm', 'theta_deg', 'vg_over_c', 'h_over_a', 'omega_D'.
        """
        h_val = float(np.clip(h_nm, self.h_min, self.h_max))
        a_val = float(self.interp_a(h_val))
        r1_val = float(self.interp_r1(h_val))
        r2_val = float(self.interp_r2(h_val))
        rho_val = float(self.interp_rho(h_val))
        theta_val = float(self.interp_theta(h_val))
        ff_val = float(self.interp_ff(h_val))
        vg_val = float(self.interp_vg(h_val))
        h_a_val = float(self.interp_h_over_a(h_val))
        omega_val = float(self.interp_omega(h_val))

        return {
            "h_nm": h_val,
            "lambda0_nm": self.lambda0_nm,
            "a_nm": a_val,
            "r1_nm": r1_val,
            "r2_nm": r2_val,
            "r1_norm": r1_val / a_val if a_val > 0 else 0.0,
            "r2_norm": r2_val / a_val if a_val > 0 else 0.0,
            "r1_over_a": r1_val / a_val if a_val > 0 else 0.0,
            "r2_over_a": r2_val / a_val if a_val > 0 else 0.0,
            "rho_nm": rho_val,
            "theta_deg": theta_val,
            "filling_factor": ff_val,
            "vg_over_c": vg_val,
            "h_over_a": h_a_val,
            "omega_D": omega_val,
        }

    def get_design_point(self, h_nm: float) -> Dict[str, float]:
        """Alias for query(h_nm)."""
        return self.query(h_nm)

    def export_dense_csv(self, output_path: Union[str, Path], n_points: int = 100) -> Path:
        """Export continuous densely sampled design curve table to CSV."""
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        h_dense = np.linspace(self.h_min, self.h_max, n_points)
        fieldnames = [
            "h_nm", "lambda0_nm", "a_nm", "r1_nm", "r2_nm",
            "r1_over_a", "r2_over_a", "rho_nm", "theta_deg",
            "filling_factor", "vg_over_c", "h_over_a", "omega_D"
        ]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for h_val in h_dense:
                writer.writerow(self.query(h_val))

        print(f"Exported design curves table ({n_points} points) to '{out_path}'")
        return out_path

    def plot_design_rules(
        self,
        output_path: Optional[Union[str, Path]] = None,
        highlight_h_nm: Optional[float] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Tuple[plt.Figure, Any]:
        """
        Generates the publication-quality multi-panel Design Rules figure
        matching Fig. 7(a) in Vallar et al., Optical Materials Express 16, 2681 (2026).
        """
        h_dense = np.linspace(self.h_min, self.h_max, 200)

        a_dense = self.interp_a(h_dense)
        ff_dense = self.interp_ff(h_dense)
        rho_dense = self.interp_rho(h_dense)
        theta_dense = self.interp_theta(h_dense)
        r1_dense = self.interp_r1(h_dense)
        r2_dense = self.interp_r2(h_dense)
        vg_dense = self.interp_vg(h_dense)

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.0, 7.2), sharex=True, dpi=dpi)

        # -------------------------------------------------------------
        # Top Panel: Filling Factor (Left) and Lattice Constant a (Right)
        # -------------------------------------------------------------
        c_ff = "#0070C0"
        c_a = "#D62728"

        # FF curve (left axis)
        l1, = ax_top.plot(h_dense, ff_dense, "-", color=c_ff, linewidth=2.4, label=r"Filling Factor $FF$")
        ax_top.scatter(self.h_nm, self.ff, color=c_ff, edgecolors="black", s=45, zorder=5)
        ax_top.set_ylabel(r"Filling Factor ($FF$)", fontsize=11, fontweight="bold", color=c_ff)
        ax_top.tick_params(axis="y", labelcolor=c_ff)
        ax_top.grid(True, linestyle=":", alpha=0.5)

        # a(nm) curve (right axis)
        ax_top_r = ax_top.twinx()
        l2, = ax_top_r.plot(h_dense, a_dense, "--", color=c_a, linewidth=2.4, label=r"Lattice Constant $a$ (nm)")
        ax_top_r.scatter(self.h_nm, self.a_nm, color=c_a, edgecolors="black", s=45, zorder=5)
        ax_top_r.set_ylabel(r"Lattice Constant $a$ (nm)", fontsize=11, fontweight="bold", color=c_a)
        ax_top_r.tick_params(axis="y", labelcolor=c_a)

        ax_top.set_title(
            rf"(a) Dirac-Like Cone Design Curves at $\lambda_0 = {int(self.lambda0_nm)}$ nm",
            fontsize=12, fontweight="bold", pad=10
        )

        # -------------------------------------------------------------
        # Bottom Panel: Polar Hole Size rho (Left) and theta (Right)
        # -------------------------------------------------------------
        c_rho = "#00B050"
        c_theta = "#7030A0"

        l3, = ax_bot.plot(h_dense, rho_dense, "-", color=c_rho, linewidth=2.4, label=r"Hole Size $\rho$ (nm)")
        ax_bot.scatter(self.h_nm, self.rho_nm, color=c_rho, edgecolors="black", s=45, zorder=5)
        ax_bot.set_ylabel(r"Hole Size $\rho = \sqrt{r_1^2 + r_2^2}$ (nm)", fontsize=11, fontweight="bold", color=c_rho)
        ax_bot.tick_params(axis="y", labelcolor=c_rho)
        ax_bot.set_xlabel("Slab Thickness $h$ (nm)", fontsize=11, fontweight="bold")
        ax_bot.grid(True, linestyle=":", alpha=0.5)

        ax_bot_r = ax_bot.twinx()
        l4, = ax_bot_r.plot(h_dense, theta_dense, ":", color=c_theta, linewidth=2.4, label=r"Hole Angle $\theta$ (deg)")
        ax_bot_r.scatter(self.h_nm, self.theta_deg, color=c_theta, edgecolors="black", s=45, zorder=5)
        ax_bot_r.set_ylabel(r"Hole Angle $\theta$ (deg)", fontsize=11, fontweight="bold", color=c_theta)
        ax_bot_r.tick_params(axis="y", labelcolor=c_theta)

        # Highlight specific slab thickness if requested
        if highlight_h_nm is not None and self.h_min <= highlight_h_nm <= self.h_max:
            h_val = float(highlight_h_nm)
            design_point = self.query(h_val)
            for ax in (ax_top, ax_bot):
                ax.axvline(h_val, color="black", linestyle="--", linewidth=1.5, alpha=0.75, zorder=4)
            ax_top.text(
                h_val + (self.h_max - self.h_min) * 0.02,
                np.mean(ax_top.get_ylim()),
                f"Target $h = {int(h_val)}$ nm\n$a = {design_point['a_nm']:.1f}$ nm\n$r_1 = {design_point['r1_nm']:.1f}$ nm\n$r_2 = {design_point['r2_nm']:.1f}$ nm",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"),
                zorder=6
            )

        fig.tight_layout()

        if output_path is not None:
            save_path = Path(output_path).resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved design rules figure to '{save_path}'")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, (ax_top, ax_bot)


def generate_design_curves(
    sweep_summary_path: Union[str, Path],
    lambda0_nm: float = 1550.0,
    output_dir: Optional[Union[str, Path]] = None,
    highlight_h_nm: Optional[float] = 375.0,
) -> SlabDesignCurves:
    """
    Main programmatic interface to build and export design curves for 3D slab.
    """
    curves = SlabDesignCurves(sweep_summary_path, lambda0_nm=lambda0_nm)
    out_dir = Path(output_dir or Path(sweep_summary_path).parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file = out_dir / f"design_rules_{int(lambda0_nm)}nm.csv"
    fig_file = out_dir / f"design_rules_{int(lambda0_nm)}nm.png"

    curves.export_dense_csv(csv_file)
    curves.plot_design_rules(output_path=fig_file, highlight_h_nm=highlight_h_nm)

    return curves


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Wavelength-Tuning Design Curves for 3D Photonic Crystal Slabs (Vallar et al., OME 2026)."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to slab sweep summary CSV / JSON file or sweep directory"
    )
    parser.add_argument(
        "--wavelength", "-w",
        type=float,
        default=1550.0,
        help="Target operating wavelength in nanometers (default: 1550.0 nm)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output image path for design rules figure (e.g. design_rules_1550nm.png)"
    )
    parser.add_argument(
        "--highlight",
        type=float,
        default=375.0,
        help="Target slab thickness h (nm) to highlight with a vertical guide line (default: 375.0 nm)"
    )

    args = parser.parse_args()
    generate_design_curves(
        sweep_summary_path=args.input,
        lambda0_nm=args.wavelength,
        output_dir=Path(args.output).parent if args.output else None,
        highlight_h_nm=args.highlight,
    )


if __name__ == "__main__":
    main()
