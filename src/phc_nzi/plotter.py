import os
import re
import argparse
import threading
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

_plot_lock = threading.Lock()


def plot_band_structure(
    data_path: Union[str, os.PathLike] = "tefreqs.data",
    labels: Optional[Union[List[str], str, os.PathLike]] = None,
    output_path: Optional[Union[str, os.PathLike]] = "band_structure.png",
    title: str = "Photonic Crystal Band Structure",
    highlight_gaps: bool = False,
    style: str = "light",
    show: bool = False,
    dpi: int = 300,
    ylim: Optional[Tuple[float, float]] = None,
    bands: Optional[List[int]] = None,
    polarization: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot photonic band structure dispersion curves from .data files.

    Parameters:
    -----------
    data_path : str or PathLike
        Path to a .data file (e.g. 'tefreqs.data') or directory containing .data files.
    labels : list of str, str, or PathLike, optional
        High-symmetry k-point labels (e.g. ['K', 'Γ', 'M']) or path to kpath_labels.data file.
    output_path : str or PathLike, optional
        Path to save figure (e.g. 'band_structure.png'). If None, figure is not saved to disk.
    title : str
        Plot title.
    highlight_gaps : bool, default False
        Whether to highlight and annotate complete photonic band gaps.
    style : {'light', 'dark'}, default 'light'
        Visual style mode.
    show : bool, default False
        Whether to call plt.show() interactively.
    dpi : int, default 300
        Output figure resolution DPI.

    Returns:
    --------
    (fig, ax) : matplotlib Figure and Axes objects.
    """
    path_obj = Path(data_path).resolve()
    target_dir = path_obj if path_obj.is_dir() else path_obj.parent

    # Mode color & label specs:
    # "te" and "zeven" -> Red ("#d62728")
    # "tm" and "zodd"  -> Blue ("#1f77b4")
    # "" (generic)    -> Black ("#000000")
    MODE_SPECS = [
        ("tefreqs.data", "#d62728", "TE Modes"),
        ("zevenfreqs.data", "#d62728", "Z-even Modes"),
        ("tmfreqs.data", "#1f77b4", "TM Modes"),
        ("zoddfreqs.data", "#1f77b4", "Z-odd Modes"),
        ("freqs.data", "#000000", "Modes"),
    ]

    files_to_plot: List[Tuple[Path, str, str]] = []

    if path_obj.is_dir():
        for candidate_filename, color, mode_label in MODE_SPECS:
            fpath = path_obj / candidate_filename
            if fpath.is_file():
                files_to_plot.append((fpath, color, mode_label))
    else:
        # User passed a specific file path. Check if other .data files exist in same dir
        fname_lower = path_obj.name.lower()
        if "te" in fname_lower or "zeven" in fname_lower:
            c, lbl = "#d62728", "TE Modes" if "te" in fname_lower else "Z-even Modes"
        elif "tm" in fname_lower or "zodd" in fname_lower:
            c, lbl = "#1f77b4", "TM Modes" if "tm" in fname_lower else "Z-odd Modes"
        else:
            c, lbl = "#000000", "Modes"

        files_to_plot.append((path_obj, c, lbl))

        # Also search for sibling .data files in target_dir if available
        for candidate_filename, color, mode_label in MODE_SPECS:
            fpath = target_dir / candidate_filename
            if fpath.is_file() and fpath != path_obj:
                files_to_plot.append((fpath, color, mode_label))

    if not files_to_plot:
        raise FileNotFoundError(f"No band structure .data files found in '{target_dir}'")

    # Resolve k-path labels
    label_list = resolve_kpath_labels(labels, target_dir)

    with _plot_lock:
        # Configure matplotlib style
        setup_plot_style(style)

        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)

        k_indices_ref: Optional[np.ndarray] = None
        all_bands_list: List[np.ndarray] = []

        for fpath, color, mode_label in files_to_plot:
            headers, data_matrix = load_data_file(fpath)
            if data_matrix.shape[0] == 0:
                continue

            k_indices = data_matrix[:, 0]
            if k_indices_ref is None:
                k_indices_ref = k_indices

            bands_data = data_matrix[:, 5:]
            num_bands = bands_data.shape[1]
            all_bands_list.append(bands_data)

            # Plot each band with no line (dots only)
            for b in range(num_bands):
                ax.plot(
                    k_indices,
                    bands_data[:, b],
                    linestyle="None",
                    marker="o",
                    markersize=4.0,
                    color=color,
                    alpha=0.85,
                    label=mode_label if b == 0 else ""  # Common legend entry per mode type
                )

        if k_indices_ref is None:
            plt.close(fig)
            raise ValueError(f"No numeric band data found in target files.")

        # Configure X-axis ticks & high-symmetry vertical lines
        if label_list and len(label_list) > 1:
            formatted_labels = [r"$\Gamma$" if l.lower() in ["gamma", "g"] else l for l in label_list]
            tick_positions = np.linspace(k_indices_ref[0], k_indices_ref[-1], len(formatted_labels))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(formatted_labels, fontsize=12, fontweight="bold")

            for pos in tick_positions:
                ax.axvline(x=pos, color="#888888" if style == "light" else "#555555", linestyle="--", linewidth=0.8, alpha=0.7)
        else:
            ax.set_xlabel("k-point Index", fontsize=11, fontweight="bold")

        # Configure Y-axis
        ax.set_ylabel(r"Frequency ($\omega a / 2\pi c$)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        # Highlight Photonic Band Gaps if requested
        if highlight_gaps and len(all_bands_list) > 0:
            combined_bands = np.hstack(all_bands_list)
            gaps = find_photonic_band_gaps(combined_bands)
            for g_idx, (gap_min, gap_max, gap_pct, lower_b) in enumerate(gaps):
                ax.axhspan(gap_min, gap_max, color="#ff7f0e", alpha=0.22, label="Band Gap" if g_idx == 0 else "")
                mid_y = (gap_min + gap_max) / 2.0
                mid_x = (k_indices_ref[0] + k_indices_ref[-1]) / 2.0
                ax.text(
                    mid_x,
                    mid_y,
                    f"Gap: {gap_pct:.1f}%",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color="#d95f02",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d95f02", alpha=0.85)
                )

        ax.set_xlim(k_indices_ref[0], k_indices_ref[-1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        elif bands is not None and len(bands) > 0 and len(all_bands_list) > 0:
            min_b = min(bands)  # 1-indexed band number (e.g. 4)
            max_b = max(bands)  # 1-indexed band number (e.g. 6)

            # Filter bands array by polarization if specified
            target_arrays = []
            for idx, (fpath, _, _) in enumerate(files_to_plot):
                fname_lower = fpath.name.lower()
                if polarization:
                    pol_lower = polarization.lower()
                    if pol_lower in ["te", "zeven"] and ("te" in fname_lower or "zeven" in fname_lower):
                        target_arrays.append(all_bands_list[idx])
                    elif pol_lower in ["tm", "zodd"] and ("tm" in fname_lower or "zodd" in fname_lower):
                        target_arrays.append(all_bands_list[idx])
                else:
                    target_arrays.append(all_bands_list[idx])
            if not target_arrays:
                target_arrays = all_bands_list

            t_min_list = []
            t_max_list = []
            for b_arr in target_arrays:
                n_cols = b_arr.shape[1]
                col_min = max(0, min_b - 1)
                col_max = min(n_cols - 1, max_b - 1)
                if col_min < n_cols:
                    t_min_list.append(np.min(b_arr[:, col_min]))
                if col_max < n_cols:
                    t_max_list.append(np.max(b_arr[:, col_max]))
            if t_min_list and t_max_list:
                ax.set_ylim(float(min(t_min_list)), float(max(t_max_list)))
            else:
                combined_bands = np.hstack(all_bands_list)
                ax.set_ylim(float(np.min(combined_bands)), float(np.max(combined_bands)))
        elif len(all_bands_list) > 0:
            combined_bands = np.hstack(all_bands_list)
            min_f = float(np.min(combined_bands))
            max_f = float(np.max(combined_bands))
            ax.set_ylim(min_f, max_f)
        else:
            ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle=":", alpha=0.5)

        ax.legend(loc="upper right", frameon=True, fontsize=9.5)

        fig.tight_layout()

        if output_path is not None:
            save_file = Path(output_path).resolve()
            save_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_file, dpi=dpi, bbox_inches="tight")
            if verbose:
                print(f"Saved band structure plot to '{save_file}'")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax


def plot_epsilon(
    h5_path: Union[str, os.PathLike],
    output_path: Optional[Union[str, os.PathLike]] = "epsilon_map.png",
    rectify: bool = True,
    options: Optional[Any] = None,
    slice_idx: Optional[int] = None,
    cmap: str = "managua_r",
    title: str = "Dielectric Function Grid (Epsilon)",
    show: bool = False,
    dpi: int = 300,
    verbose: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D dielectric epsilon cross-section map from an HDF5 grid file.

    Parameters:
    -----------
    h5_path : str or PathLike
        Path to HDF5 epsilon file (e.g. 'main-epsilon.h5').
    output_path : str or PathLike, optional
        Path to save image figure.
    rectify : bool, default True
        Whether to run mpb-data to geometrically rectify non-orthogonal/hexagonal grids to Cartesian coordinates.
    options : MPBDataOptions, optional
        Options for mpb-data conversion (default: MPBDataOptions(rectify=True, periods=(3, 3, 1))).
    slice_idx : int, optional
        Z-plane slice index for 3D grids. Defaults to center slice.
    cmap : str, default 'viridis'
        Matplotlib colormap for dielectric values.
    title : str
        Plot title.
    show : bool, default False
        Whether to call plt.show().
    verbose : bool, default False
        Whether to print status messages.

    Returns:
    --------
    (fig, ax) : matplotlib Figure and Axes objects.
    """
    h5_file = Path(h5_path).resolve()
    if not h5_file.is_file():
        raise FileNotFoundError(f"HDF5 file not found at '{h5_file}'")

    if rectify and not h5_file.name.endswith(".converted.h5"):
        try:
            from .transformer import transform_h5_data, MPBDataOptions
            h5_file = transform_h5_data(h5_file, options=options or MPBDataOptions(), verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"Note: Grid transformation fallback: {e}")

    with h5py.File(h5_file, "r") as f:
        key = "data" if "data" in f else ("epsilon.xx" if "epsilon.xx" in f else list(f.keys())[0])
        data = np.array(f[key])

    # Extract 2D slice if 3D array
    if data.ndim == 3:
        if slice_idx is None:
            slice_idx = data.shape[2] // 2
        data_2d = data[:, :, slice_idx]
    elif data.ndim == 2:
        data_2d = data
    else:
        raise ValueError(f"Unsupported array dimensions ({data.ndim}D) in HDF5 file")

    with _plot_lock:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        im = ax.imshow(data_2d.T, origin="lower", cmap=cmap, interpolation="nearest")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Permittivity $\epsilon$", fontsize=11, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Grid X", fontsize=10, fontweight="bold")
        ax.set_ylabel("Grid Y", fontsize=10, fontweight="bold")
        fig.tight_layout()

        if output_path is not None:
            save_file = Path(output_path).resolve()
            save_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_file, dpi=dpi, bbox_inches="tight")
            if verbose:
                print(f"Saved dielectric epsilon plot to '{save_file}'")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax


def load_data_file(filepath: Path) -> Tuple[List[str], np.ndarray]:
    """Parse .data space-delimited text file into header list and numpy array."""
    lines = filepath.read_text().splitlines()
    headers: List[str] = []
    rows: List[List[float]] = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.startswith("#"):
            headers = line_str.lstrip("#").strip().split()
            continue
        parts = line_str.split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue

    return headers, np.array(rows) if rows else np.empty((0, 0))


def resolve_kpath_labels(
    labels_arg: Optional[Union[List[str], str, os.PathLike]],
    target_dir: Path
) -> Optional[List[str]]:
    """Resolve k-path label list from arguments or file."""
    if isinstance(labels_arg, list):
        return labels_arg

    if isinstance(labels_arg, (str, os.PathLike)):
        lbl_file = Path(labels_arg)
        if lbl_file.is_file():
            return lbl_file.read_text().strip().split()

    default_lbl_file = target_dir / "kpath_labels.data"
    if default_lbl_file.is_file():
        return default_lbl_file.read_text().strip().split()

    return None


def find_photonic_band_gaps(bands_data: np.ndarray) -> List[Tuple[float, float, float, int]]:
    """Find complete photonic band gaps across all k-points."""
    num_kpts, num_bands = bands_data.shape
    gaps = []

    for b in range(num_bands - 1):
        max_lower_band = np.max(bands_data[:, b])
        min_upper_band = np.min(bands_data[:, b + 1])

        if min_upper_band > max_lower_band:
            gap_mid = (min_upper_band + max_lower_band) / 2.0
            gap_size = min_upper_band - max_lower_band
            gap_pct = (gap_size / gap_mid) * 100.0
            gaps.append((max_lower_band, min_upper_band, gap_pct, b + 1))

    return gaps


def setup_plot_style(style: str) -> None:
    """Configure publication-ready font and style defaults."""
    if style == "dark":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")


def main() -> None:
    """CLI entry point for plotter tool."""
    parser = argparse.ArgumentParser(
        description="Plot photonic band structures and dielectric grids from MPB simulations."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="tefreqs.data",
        help="Input .data file path or directory (default: tefreqs.data)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="band_structure.png",
        help="Output image path (default: band_structure.png)"
    )
    parser.add_argument(
        "-l", "--labels",
        type=str,
        default=None,
        help="Path to kpath_labels.data or space-separated labels string"
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default="Photonic Band Structure",
        help="Plot title"
    )
    parser.add_argument(
        "--h5",
        type=str,
        default=None,
        help="Path to HDF5 file (e.g. example-epsilon.h5) to plot 2D dielectric grid instead"
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Use dark mode theme for plot"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively"
    )

    args = parser.parse_args()

    style = "dark" if args.dark else "light"

    if args.h5:
        plot_epsilon(
            h5_path=args.h5,
            output_path=args.output,
            title=args.title,
            show=args.show
        )
    else:
        labels_input = args.labels.split() if args.labels and " " in args.labels else args.labels
        plot_band_structure(
            data_path=args.input,
            labels=labels_input,
            output_path=args.output,
            title=args.title,
            style=style,
            show=args.show
        )


if __name__ == "__main__":
    main()
