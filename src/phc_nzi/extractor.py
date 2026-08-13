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
        Dictionary mapping polarization keys ('tefreqs', 'tmfreqs', 'zevenfreqs', 'zoddfreqs', 'freqs')
        to parsed headers and rows.
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
    polarization_types = ["tefreqs", "tmfreqs", "zevenfreqs", "zoddfreqs", "freqs"]

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
                print(f"Extracted {len(data_rows)} k-points for '{pol}' to '{data_filepath}'")

    # Extract KPATH_LABELS if present in the log
    labels = extract_kpath_labels(lines)
    if labels:
        extracted_data["kpath_labels"] = labels
        if save_data:
            labels_filepath = target_dir / "kpath_labels.data"
            labels_filepath.write_text(" ".join(labels) + "\n")
            print(f"Extracted k-path labels {labels} to '{labels_filepath}'")

    # Extract symmetry / irrep data if present in log
    full_text = out_file.read_text()
    if "SYM_DATA_START_" in full_text:
        sym_records = extract_symmetries(
            output_path=out_file,
            output_dir=target_dir,
            save_data=save_data,
            freq_data=extracted_data
        )
        if sym_records:
            extracted_data["symmetries"] = sym_records

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


def extract_symmetries(
    output_path: Union[str, os.PathLike] = "output.out",
    output_dir: Optional[Union[str, os.PathLike]] = None,
    save_data: bool = True,
    freq_data: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Extract symmetry characters and irrep classifications at Gamma from MPB log files.

    Parameters:
    -----------
    output_path : str or PathLike
        Path to MPB simulation log file.
    output_dir : str or PathLike, optional
        Target directory to save symmetries.data and symmetries.json.
    save_data : bool, default True
        Whether to save extracted symmetries to disk.
    freq_data : dict, optional
        Extracted frequency data for frequency correlation.

    Returns:
    --------
    list of dict
        List of irrep classification dictionaries per band and parity.
    """
    from .symmetry import analyze_symmetries_from_log

    out_file = Path(output_path).resolve()
    if not out_file.is_file():
        raise FileNotFoundError(f"MPB log file not found at '{out_file}'")

    if output_dir is None:
        target_dir = out_file.parent
    else:
        target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    text = out_file.read_text()
    records = analyze_symmetries_from_log(text, freq_data=freq_data)

    if records and save_data:
        save_symmetries(records, target_dir)

    return records


def save_symmetries(records: List[Dict[str, Any]], target_dir: Path) -> None:
    """Save extracted symmetry records to symmetries.data and symmetries.json files."""
    import json

    # 1. Plain text table format (symmetries.data)
    data_filepath = target_dir / "symmetries.data"
    headers = ["parity", "band", "freq", "irrep", "confidence", "point_group", "k_index", "k1", "k2", "k3", "kmag_2pi"]

    with open(data_filepath, "w") as f:
        f.write("# " + " ".join(headers) + "\n")
        for r in records:
            freq_str = f"{r['freq']:.8g}" if isinstance(r['freq'], (int, float)) and not math_isnan(r['freq']) else "nan"
            k_idx_str = str(r.get("k_index")) if r.get("k_index") is not None else "nan"
            k1_val = r.get("k1", 0.0)
            k2_val = r.get("k2", 0.0)
            k3_val = r.get("k3", 0.0)
            kmag_val = r.get("kmag_2pi", 0.0)
            f.write(f"{r['parity']} {r['band']} {freq_str} {r['irrep']} {r['confidence']:.4f} {r['point_group']} {k_idx_str} {k1_val:.6g} {k2_val:.6g} {k3_val:.6g} {kmag_val:.6g}\n")

    print(f"Extracted {len(records)} band symmetry irreps to '{data_filepath}'")

    # 2. JSON format (symmetries.json) with full character and projection details
    json_filepath = target_dir / "symmetries.json"
    json_filepath.write_text(json.dumps(records, indent=2))
    print(f"Saved detailed symmetry JSON to '{json_filepath}'")


def load_symmetries(symmetries_filepath: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
    """
    Load extracted symmetries from a symmetries.data or symmetries.json file.
    """
    import json

    filepath = Path(symmetries_filepath).resolve()
    if not filepath.is_file():
        raise FileNotFoundError(f"Symmetry data file not found at '{filepath}'")

    if filepath.suffix == ".json":
        return json.loads(filepath.read_text())

    records: List[Dict[str, Any]] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    freq_val = float(parts[2]) if parts[2] != "nan" else float("nan")
                    k_idx_val = int(parts[6]) if len(parts) > 6 and parts[6] != "nan" else None
                    k1_val = float(parts[7]) if len(parts) > 7 else 0.0
                    k2_val = float(parts[8]) if len(parts) > 8 else 0.0
                    k3_val = float(parts[9]) if len(parts) > 9 else 0.0
                    kmag_val = float(parts[10]) if len(parts) > 10 else 0.0

                    records.append({
                        "parity": parts[0],
                        "band": int(parts[1]),
                        "freq": freq_val,
                        "irrep": parts[3],
                        "confidence": float(parts[4]),
                        "point_group": parts[5],
                        "k_index": k_idx_val,
                        "k1": k1_val,
                        "k2": k2_val,
                        "k3": k3_val,
                        "kmag_2pi": kmag_val,
                    })
                except ValueError:
                    continue
    return records


def math_isnan(val: Any) -> bool:
    """Helper to check if float is NaN."""
    try:
        import math
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False


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
