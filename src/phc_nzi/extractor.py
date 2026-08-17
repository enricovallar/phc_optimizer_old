import os
import re
import argparse
from pathlib import Path
from typing import Union, Dict, List, Optional, Any


def extract_frequencies(
    output_path: Union[str, os.PathLike] = "output.out",
    output_dir: Optional[Union[str, os.PathLike]] = None,
    save_data: bool = True,
    verbose: bool = True,
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
                if verbose:
                    print(f"Extracted {len(data_rows)} k-points for '{pol}' to '{data_filepath}'")

    # Extract KPATH_LABELS if present in the log
    labels = extract_kpath_labels(lines)
    if labels:
        extracted_data["kpath_labels"] = labels
        if save_data:
            labels_filepath = target_dir / "kpath_labels.data"
            labels_filepath.write_text(" ".join(labels) + "\n")
            if verbose:
                print(f"Extracted k-path labels {labels} to '{labels_filepath}'")

    # Extract symmetry / irrep data if present in log
    full_text = out_file.read_text()
    if "SYM_DATA_START_" in full_text:
        sym_records = extract_symmetries(
            output_path=out_file,
            output_dir=target_dir,
            save_data=save_data,
            freq_data=extracted_data,
            verbose=verbose,
        )

        if sym_records:
            extracted_data["symmetries"] = sym_records

    # Extract group velocity data if present in log
    if "velocity:" in full_text:
        vel_records = extract_group_velocities(
            output_path=out_file,
            output_dir=target_dir,
            save_data=save_data,
            freq_data=extracted_data,
            verbose=verbose
        )
        if vel_records:
            extracted_data["group_velocities"] = vel_records

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
    freq_data: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
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
    Extract symmetry character expectation values and project onto irreps.
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
        save_symmetries(records, target_dir, verbose=verbose)

    return records


def save_symmetries(records: List[Dict[str, Any]], target_dir: Path, verbose: bool = True) -> None:
    """Save extracted symmetry records to symmetries.json file."""
    import json

    json_filepath = target_dir / "symmetries.json"
    json_filepath.write_text(json.dumps(records, indent=2))
    if verbose:
        print(f"Extracted {len(records)} band symmetry irreps to '{json_filepath}'")


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


def extract_group_velocities(
    output_path: Union[str, os.PathLike] = "output.out",
    output_dir: Optional[Union[str, os.PathLike]] = None,
    save_data: bool = True,
    freq_data: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Extract group velocity vectors and magnitudes from MPB log files (output.out).

    Parameters:
    -----------
    output_path : str or PathLike
        Path to MPB simulation log file.
    output_dir : str or PathLike, optional
        Target directory to save .data files (group_velocities.data, tevelocity.data, etc.).
    save_data : bool, default True
        Whether to save extracted group velocity data files to disk.
    freq_data : dict, optional
        Extracted frequency data for k-vector coordinate alignment.
    verbose : bool, default True
        Whether to print extraction progress to console.

    Returns:
    --------
    dict
        Dictionary mapping polarization keys ('tevelocity', 'tmvelocity', 'flat_records')
        to parsed rows and vectors.
    """
    import math

    out_file = Path(output_path).resolve()
    if not out_file.is_file():
        raise FileNotFoundError(f"MPB log file not found at '{out_file}'")

    if output_dir is None:
        target_dir = out_file.parent
    else:
        target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    lines = out_file.read_text().splitlines()
    polarization_types = ["te", "tm", "zeven", "zodd"]

    results: Dict[str, Any] = {}
    flat_records: List[Dict[str, Any]] = []

    for pol in polarization_types:
        prefix = f"{pol}velocity:"
        k_map: Dict[int, Dict[int, Tuple[float, float, float]]] = {}

        for line in lines:
            line_str = line.strip()
            if not line_str.startswith(prefix):
                continue

            parts = [p.strip() for p in line_str.split(",") if p.strip()]
            if len(parts) < 3:
                continue

            try:
                k_idx = int(parts[1])
            except ValueError:
                continue

            band_vecs: Dict[int, Tuple[float, float, float]] = {}
            for col_i, item in enumerate(parts[2:], start=1):
                clean_vec = item.replace("#(", "").replace("(", "").replace(")", "").strip()
                v_parts = clean_vec.split()
                if len(v_parts) >= 3:
                    try:
                        vx = float(v_parts[0])
                        vy = float(v_parts[1])
                        vz = float(v_parts[2])
                        band_vecs[col_i] = (vx, vy, vz)
                    except ValueError:
                        pass
                elif len(v_parts) == 1:
                    try:
                        vx = float(v_parts[0])
                        band_vecs[col_i] = (vx, 0.0, 0.0)
                    except ValueError:
                        pass

            if band_vecs:
                k_map[k_idx] = band_vecs

        if not k_map:
            continue

        # Look up k-vector coordinates from freq_data if available, or parse from log lines directly
        k_coords_lookup: Dict[int, Tuple[float, float, float, float]] = {}
        pol_freq_key = f"{pol}freqs"
        if freq_data and pol_freq_key in freq_data:
            rows = freq_data[pol_freq_key].get("rows", [])
            headers = freq_data[pol_freq_key].get("headers", [])
            if rows and headers:
                k1_i = headers.index("k1") if "k1" in headers else 1
                k2_i = headers.index("k2") if "k2" in headers else 2
                k3_i = headers.index("k3") if "k3" in headers else 3
                kmag_i = next((i for i, h in enumerate(headers) if "kmag" in h), None)
                k_idx_i = headers.index("k_index") if "k_index" in headers else 0

                for r in rows:
                    try:
                        kidx_val = int(r[k_idx_i])
                        k1_val = float(r[k1_i])
                        k2_val = float(r[k2_i])
                        k3_val = float(r[k3_i])
                        kmag_val = float(r[kmag_i]) if kmag_i is not None else math.sqrt(k1_val**2 + k2_val**2 + k3_val**2)
                        k_coords_lookup[kidx_val] = (k1_val, k2_val, k3_val, kmag_val)
                    except (ValueError, IndexError):
                        pass

        if not k_coords_lookup:
            freq_prefix = f"{pol}freqs:"
            for line in lines:
                l_str = line.strip()
                if l_str.startswith(freq_prefix):
                    p_parts = [p.strip() for p in l_str.split(",") if p.strip()]
                    if len(p_parts) >= 6:
                        try:
                            kidx_val = int(p_parts[1])
                            k1_val = float(p_parts[2])
                            k2_val = float(p_parts[3])
                            k3_val = float(p_parts[4])
                            kmag_val = float(p_parts[5])
                            k_coords_lookup[kidx_val] = (k1_val, k2_val, k3_val, kmag_val)
                        except (ValueError, IndexError):
                            pass

        # Build matrix rows and flat records
        matrix_rows: List[List[Union[int, float]]] = []
        max_bands = max(max(b_dict.keys()) for b_dict in k_map.values()) if k_map else 0

        headers = ["k_index", "k1", "k2", "k3", "kmag_2pi"]
        for b in range(1, max_bands + 1):
            headers.extend([f"band_{b}_vx", f"band_{b}_vy", f"band_{b}_vz", f"band_{b}_vg"])

        for k_idx in sorted(k_map.keys()):
            k1, k2, k3, kmag = k_coords_lookup.get(k_idx, (0.0, 0.0, 0.0, 0.0))
            row_items: List[Union[int, float]] = [k_idx, k1, k2, k3, kmag]

            for b_idx in range(1, max_bands + 1):
                vec = k_map[k_idx].get(b_idx, (float("nan"), float("nan"), float("nan")))
                vx, vy, vz = vec
                if not (math.isnan(vx) or math.isnan(vy) or math.isnan(vz)):
                    vg_mag = math.sqrt(vx**2 + vy**2 + vz**2)
                else:
                    vg_mag = float("nan")

                row_items.extend([vx, vy, vz, vg_mag])

                if not math.isnan(vg_mag):
                    flat_records.append({
                        "parity": pol,
                        "band": b_idx,
                        "k_index": k_idx,
                        "k1": k1,
                        "k2": k2,
                        "k3": k3,
                        "kmag_2pi": kmag,
                        "vx": vx,
                        "vy": vy,
                        "vz": vz,
                        "vg_mag": vg_mag
                    })

            matrix_rows.append(row_items)

        pol_vel_key = f"{pol}velocity"
        results[pol_vel_key] = {
            "headers": headers,
            "rows": matrix_rows
        }

        if save_data:
            data_filepath = target_dir / f"{pol_vel_key}.data"
            write_data_file(headers, matrix_rows, data_filepath)
            if verbose:
                print(f"Extracted group velocities for '{pol_vel_key}' to '{data_filepath}'")

    results["flat_records"] = flat_records

    if save_data and flat_records:
        save_group_velocities(flat_records, target_dir, verbose=verbose)

    return results


def save_group_velocities(records: List[Dict[str, Any]], target_dir: Path, verbose: bool = True) -> None:
    """Save group velocity records to group_velocities.json file."""
    import json

    json_filepath = target_dir / "group_velocities.json"
    json_filepath.write_text(json.dumps(records, indent=2))
    if verbose:
        print(f"Extracted {len(records)} group velocity records to '{json_filepath}'")


def load_group_velocities(velocities_filepath: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
    """
    Load extracted group velocities from a group_velocities.data or group_velocities.json file.
    """
    import json

    filepath = Path(velocities_filepath).resolve()
    if not filepath.is_file():
        raise FileNotFoundError(f"Group velocity data file not found at '{filepath}'")

    if filepath.suffix == ".json":
        return json.loads(filepath.read_text())

    records: List[Dict[str, Any]] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 11:
                try:
                    records.append({
                        "parity": parts[0],
                        "band": int(parts[1]),
                        "k_index": int(parts[2]),
                        "k1": float(parts[3]),
                        "k2": float(parts[4]),
                        "k3": float(parts[5]),
                        "kmag_2pi": float(parts[6]),
                        "vx": float(parts[7]),
                        "vy": float(parts[8]),
                        "vz": float(parts[9]),
                        "vg_mag": float(parts[10]),
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
