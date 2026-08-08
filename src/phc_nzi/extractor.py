import os
import re
import argparse
from pathlib import Path
from typing import Union, Dict, List, Optional, Any


def extract_frequencies(
    output_path: Union[str, os.PathLike] = "output.out",
    output_dir: Optional[Union[str, os.PathLike]] = None,
    save_data: bool = True
) -> Dict[str, Any]:
    """
    Extract frequency band data and k-path information from MPB log files (output.out).

    Parameters:
    -----------
    output_path : str or PathLike
        Path to MPB simulation log file (e.g., 'output.out' or 'work/output.out').
    output_dir : str or PathLike, optional
        Target directory to save extracted .data files. Defaults to same directory as output_path.
    save_data : bool, default True
        Whether to save extracted frequencies to .data text files.

    Returns:
    --------
    dict
        Dictionary mapping polarization keys ('tefreqs', 'tmfreqs', 'freqs') to parsed headers and rows.
    """
    out_file = Path(output_path).resolve()
    if not out_file.is_file():
        raise FileNotFoundError(f"MPB output log file not found at '{out_file}'")

    if output_dir is None:
        target_dir = out_file.parent
    else:
        target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    lines = out_file.read_text().splitlines()

    extracted_data: Dict[str, Any] = {}
    polarization_types = ["tefreqs", "tmfreqs", "freqs"]

    for pol in polarization_types:
        header_cols: Optional[List[str]] = None
        data_rows: List[List[Union[int, float]]] = []

        for line in lines:
            if not line.startswith(f"{pol}:"):
                continue

            parts = [p.strip() for p in line.split(",")[1:] if p.strip()]
            if not parts:
                continue

            # Check if line is header (contains column names)
            if header_cols is None and any(not is_float(p) for p in parts):
                header_cols = [clean_col_name(p) for p in parts]
                continue

            # Parse numeric data row
            if all(is_float(p) for p in parts):
                row_vals: List[Union[int, float]] = []
                for idx, p in enumerate(parts):
                    val = float(p)
                    if idx == 0:
                        row_vals.append(int(val))
                    else:
                        row_vals.append(val)
                data_rows.append(row_vals)

        if data_rows:
            if header_cols is None or len(header_cols) != len(data_rows[0]):
                num_cols = len(data_rows[0])
                header_cols = ["k_index", "k1", "k2", "k3", "kmag_2pi"] + [
                    f"band_{i+1}" for i in range(max(0, num_cols - 5))
                ]

            extracted_data[pol] = {
                "headers": header_cols,
                "rows": data_rows
            }

            if save_data:
                data_filepath = target_dir / f"{pol}.data"
                write_data_file(header_cols, data_rows, data_filepath)
                print(f"Extracted {len(data_rows)} k-points to '{data_filepath}'")

    # Extract KPATH_LABELS if present in the log
    labels = extract_kpath_labels(lines)
    if labels:
        extracted_data["kpath_labels"] = labels
        if save_data:
            labels_filepath = target_dir / "kpath_labels.data"
            labels_filepath.write_text(" ".join(labels) + "\n")
            print(f"Extracted k-path labels {labels} to '{labels_filepath}'")

    return extracted_data


def is_float(val: str) -> bool:
    """Check if string can be converted to float."""
    try:
        float(val)
        return True
    except ValueError:
        return False


def clean_col_name(col: str) -> str:
    """Clean column name for header formatting."""
    col = col.lower().replace("/", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", col)


def write_data_file(headers: List[str], rows: List[List[Union[int, float]]], filepath: Path) -> None:
    """Write formatted columns and rows to a space-delimited .data text file."""
    with open(filepath, "w") as f:
        # Header line
        f.write("# " + " ".join(headers) + "\n")
        # Data rows
        for row in rows:
            formatted_items = []
            for item in row:
                if isinstance(item, int):
                    formatted_items.append(f"{item:d}")
                else:
                    formatted_items.append(f"{item:.8g}")
            f.write(" ".join(formatted_items) + "\n")


def extract_kpath_labels(lines: List[str]) -> Optional[List[str]]:
    """Extract KPATH_LABELS if logged in output."""
    for line in lines:
        if "KPATH_LABELS:" in line:
            match = re.search(r"KPATH_LABELS:\s*\((.*?)\)", line)
            if match:
                return match.group(1).split()
            parts = line.split("KPATH_LABELS:")[1].strip().split()
            return parts
    return None


def main() -> None:
    """CLI entry point for data extractor tool."""
    parser = argparse.ArgumentParser(
        description="Extract MPB frequency bands from log files into .data files."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="output.out",
        help="Input MPB log file path (default: output.out)"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default=None,
        help="Target directory to save .data files (default: same directory as input file)"
    )

    args = parser.parse_args()
    extract_frequencies(output_path=args.input, output_dir=args.outdir, save_data=True)


if __name__ == "__main__":
    main()
