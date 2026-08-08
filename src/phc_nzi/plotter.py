import os
import re
import argparse
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_band_structure(
    data_path: Union[str, os.PathLike] = "tefreqs.data",
    labels: Optional[Union[List[str], str, os.PathLike]] = None,
    output_path: Optional[Union[str, os.PathLike]] = "band_structure.png",
    title: str = "Photonic Crystal Band Structure",
    highlight_gaps: bool = False,
    style: str = "light",
    show: bool = False,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot photonic band structure dispersion curves from a .data file.

    Parameters:
    -----------
    data_path : str or PathLike
        Path to .data file (e.g. 'tefreqs.data') or directory containing .data files.
    labels : list of str, str, or PathLike, optional
        High-symmetry k-point labels (e.g. ['K', 'Γ', 'M']) or path to kpath_labels.data file.
    output_path : str or PathLike, optional
        Path to save figure (e.g. 'band_structure.png'). If None, figure is not saved to disk.
    title : str
        Plot title.
    highlight_gaps : bool, default True
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
    if path_obj.is_dir():
        # Look for tefreqs.data, tmfreqs.data, or freqs.data in directory
        for candidate in ["tefreqs.data", "tmfreqs.data", "freqs.data"]:
            if (path_obj / candidate).is_file():
                path_obj = path_obj / candidate
                break

    if not path_obj.is_file():
        raise FileNotFoundError(f"Band structure data file not found at '{path_obj}'")

    headers, data_matrix = load_data_file(path_obj)
    if data_matrix.shape[0] == 0:
        raise ValueError(f"No numeric rows found in data file '{path_obj}'")

    # Determine polarization prefix from filename or headers
    pol_name = "TE" if "te" in path_obj.name.lower() else ("TM" if "tm" in path_obj.name.lower() else "Band")

    # Column 0 is k_index, column 4 is kmag_2pi, columns 5+ are band frequencies
    k_indices = data_matrix[:, 0]
    bands_data = data_matrix[:, 5:]
    num_bands = bands_data.shape[1]

    # Resolve k-path labels
    label_list = resolve_kpath_labels(labels, path_obj.parent)

    # Configure matplotlib style
    setup_plot_style(style)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)

    # Plot band curves
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_bands)))
    for b in range(num_bands):
        ax.plot(
            k_indices,
            bands_data[:, b],
            label=f"Band {b+1}",
            color=colors[b % len(colors)],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            alpha=0.9
        )

    # Configure X-axis ticks & high-symmetry vertical lines
    if label_list and len(label_list) > 1:
        formatted_labels = [r"$\Gamma$" if l.lower() in ["gamma", "g"] else l for l in label_list]
        tick_positions = np.linspace(k_indices[0], k_indices[-1], len(formatted_labels))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(formatted_labels, fontsize=12, fontweight="bold")

        for pos in tick_positions:
            ax.axvline(x=pos, color="#888888" if style == "light" else "#555555", linestyle="--", linewidth=0.8, alpha=0.7)
    else:
        ax.set_xlabel("k-point Index", fontsize=11, fontweight="bold")

    # Configure Y-axis
    ax.set_ylabel(r"Frequency ($\omega a / 2\pi c$)", fontsize=12, fontweight="bold")
    ax.set_title(f"{title} ({pol_name} Modes)", fontsize=13, fontweight="bold", pad=12)

    # Highlight Photonic Band Gaps if requested
    if highlight_gaps and num_bands > 1:
        gaps = find_photonic_band_gaps(bands_data)
        for g_idx, (gap_min, gap_max, gap_pct, lower_b) in enumerate(gaps):
            ax.axhspan(gap_min, gap_max, color="#ff7f0e", alpha=0.22, label="Band Gap" if g_idx == 0 else "")
            mid_y = (gap_min + gap_max) / 2.0
            mid_x = (k_indices[0] + k_indices[-1]) / 2.0
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

    ax.set_xlim(k_indices[0], k_indices[-1])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=":", alpha=0.5)

    if num_bands <= 10:
        ax.legend(loc="upper right", frameon=True, fontsize=8.5, ncol=2 if num_bands > 5 else 1)

    fig.tight_layout()

    if output_path is not None:
        save_file = Path(output_path).resolve()
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=dpi, bbox_inches="tight")
        print(f"Saved band structure plot to '{save_file}'")

    if show:
        plt.show()

    return fig, ax


def plot_epsilon(
    h5_path: Union[str, os.PathLike],
    output_path: Optional[Union[str, os.PathLike]] = "epsilon_map.png",
    rectify: bool = True,
    options: Optional[Any] = None,
    slice_idx: Optional[int] = None,
    cmap: str = "viridis",
    title: str = "Dielectric Function Grid (Epsilon)",
    show: bool = False,
    dpi: int = 300,
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

    Returns:
    --------
    (fig, ax) : matplotlib Figure and Axes objects.
    """
    h5_file = Path(h5_path).resolve()
    if not h5_file.is_file():
        raise FileNotFoundError(f"HDF5 file not found at '{h5_file}'")

    if rectify and not h5_file.name.endswith(".converted.h5"):
        try:
            from .rectifier import rectify_h5_data, MPBDataOptions
            h5_file = rectify_h5_data(h5_file, options=options or MPBDataOptions())
        except Exception as e:
            print(f"Note: Grid rectification fallback: {e}")

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
        print(f"Saved dielectric epsilon plot to '{save_file}'")

    if show:
        plt.show()

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
