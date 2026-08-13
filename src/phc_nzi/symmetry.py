import re
from typing import Dict, List, Optional, Tuple, Any, Union


# Point Group Character Tables
# Weights correspond to the number of symmetry operations in each conjugacy class.
CHARACTER_TABLES: Dict[str, Dict[str, Any]] = {
    "C4v": {
        "g": 8,
        "weights": {"E": 1, "C4": 2, "C2": 1, "sv": 2, "sd": 2},
        "irreps": {
            "A_1": {"E": 1, "C4": 1, "C2": 1, "sv": 1, "sd": 1},
            "A_2": {"E": 1, "C4": 1, "C2": 1, "sv": -1, "sd": -1},
            "B_1": {"E": 1, "C4": -1, "C2": 1, "sv": 1, "sd": -1},
            "B_2": {"E": 1, "C4": -1, "C2": 1, "sv": -1, "sd": 1},
            "E": {"E": 2, "C4": 0, "C2": -2, "sv": 0, "sd": 0},
        },
    },
    "C6v": {
        "g": 12,
        "weights": {"E": 1, "C6": 2, "C3": 2, "C2": 1, "sv": 3, "sd": 3},
        "irreps": {
            "A_1": {"E": 1, "C6": 1, "C3": 1, "C2": 1, "sv": 1, "sd": 1},
            "A_2": {"E": 1, "C6": 1, "C3": 1, "C2": 1, "sv": -1, "sd": -1},
            "B_1": {"E": 1, "C6": -1, "C3": 1, "C2": -1, "sv": 1, "sd": -1},
            "B_2": {"E": 1, "C6": -1, "C3": 1, "C2": -1, "sv": -1, "sd": 1},
            "E_1": {"E": 2, "C6": 1, "C3": -1, "C2": -2, "sv": 0, "sd": 0},
            "E_2": {"E": 2, "C6": -1, "C3": -1, "C2": 2, "sv": 0, "sd": 0},
        },
    },
}


def detect_point_group(operators: Union[List[str], Dict[str, Any]]) -> str:
    """
    Automatically detect the point group (C4v vs C6v) based on operators present in symmetry data.

    Parameters:
    -----------
    operators : list of str or dict keys
        Names of symmetry operators parsed from output (e.g. ['C4', 'C2', 'sv', 'sd']).

    Returns:
    --------
    str
        Detected point group name ('C4v' or 'C6v').
    """
    ops = set(operators)
    if "C6" in ops or "C3" in ops:
        return "C6v"
    elif "C4" in ops:
        return "C4v"
    # Fallback default if ops unclear
    return "C4v"


def parse_complex(val_str: str) -> complex:
    """
    Safely parse MPB character string into a Python complex number.
    Handles formats like '1.0', '-1.0', '0.5+0.866i', '-0.5 - 0.866i'.
    """
    cleaned = (
        val_str.strip()
        .replace(" ", "")
        .replace("i", "j")
        .replace("I", "j")
    )
    try:
        return complex(cleaned)
    except ValueError:
        return complex(0.0)


def compute_projections(
    chars: Dict[str, complex], group: str = "C4v"
) -> Dict[str, float]:
    """
    Computes projection scalar products onto each irreducible representation of point group.

    Parameters:
    -----------
    chars : dict
        Observed character expectations per operator (e.g. {'C4': 1.0, 'C2': 1.0, ...}).
    group : str
        Point group ('C4v' or 'C6v').

    Returns:
    --------
    dict
        Projection scores for each irrep (e.g. {'A_1': 1.0, 'A_2': 0.0, ...}).
    """
    if group not in CHARACTER_TABLES:
        raise ValueError(f"Unsupported point group '{group}'. Supported groups: {list(CHARACTER_TABLES.keys())}")

    group_data = CHARACTER_TABLES[group]
    g = group_data["g"]
    weights = group_data["weights"]
    irreps = group_data["irreps"]

    # Observed Identity E is 1.0 for a single band state
    obs_chars = chars.copy()
    obs_chars["E"] = complex(1.0, 0.0)

    projections: Dict[str, float] = {}

    for irrep, irrep_chars in irreps.items():
        d_i = irrep_chars["E"]  # Irrep dimension
        scalar_product = 0.0 + 0.0j

        for op, w in weights.items():
            if op in obs_chars:
                chi_target = irrep_chars[op]
                chi_obs = obs_chars[op]
                scalar_product += w * (chi_target * chi_obs)

        # Projection formula for single state: a_i = d_i * <chi_irrep, chi_obs> / g
        proj_val = (d_i * scalar_product / g).real
        projections[irrep] = float(max(0.0, proj_val))

    return projections


def identify_irrep(projections: Dict[str, float]) -> Tuple[str, float]:
    """
    Identifies the best-matching irreducible representation label and confidence score.

    Returns:
    --------
    tuple (best_irrep, confidence_score)
    """
    if not projections:
        return "Unknown", 0.0

    best_irrep = max(projections, key=projections.get)  # type: ignore
    confidence = projections[best_irrep]
    return best_irrep, confidence


def parse_symmetry_blocks(output_text: str) -> Dict[str, Dict[int, Dict[str, complex]]]:
    """
    Extracts raw symmetry data blocks from MPB log text per parity (e.g. 'te', 'tm', 'zeven', 'zodd').

    Returns:
    --------
    dict mapping parity string to dict of {band_number: {operator_name: character_complex}}
    """
    results: Dict[str, Dict[int, Dict[str, complex]]] = {}

    # Matches SYM_DATA_START_<parity> ... SYM_DATA_END_<parity>
    pattern = r"SYM_DATA_START_(\w+)\s*\n(.*?)\n\s*SYM_DATA_END_\1"
    matches = re.findall(pattern, output_text, re.DOTALL)

    for parity, block_text in matches:
        parity_lower = parity.lower()
        band_data: Dict[int, Dict[str, complex]] = {}

        for line in block_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 3:
                continue

            try:
                band_num = int(parts[1])
            except ValueError:
                continue

            chars: Dict[str, complex] = {}
            for op_val in parts[2:]:
                if "=" not in op_val:
                    continue
                op, val = op_val.split("=", 1)
                chars[op.strip()] = parse_complex(val)

            if chars:
                band_data[band_num] = chars

        if band_data:
            results[parity_lower] = band_data

    return results


def analyze_symmetries_from_log(
    output_text: str,
    freq_data: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Analyzes MPB log text, parses symmetry blocks, computes irrep projections, and returns
    structured records.

    Parameters:
    -----------
    output_text : str
        Full MPB output log text.
    freq_data : dict, optional
        Extracted frequency data (from extractor.extract_frequencies) to correlate band frequencies at Gamma.

    Returns:
    --------
    list of dict
        List of dictionaries with keys:
        ['parity', 'band', 'freq', 'irrep', 'confidence', 'point_group', 'characters']
    """
    parsed_blocks = parse_symmetry_blocks(output_text)
    records: List[Dict[str, Any]] = []

    for parity, band_map in parsed_blocks.items():
        if not band_map:
            continue

        # Detect point group from first band's operators
        first_band_ops = list(next(iter(band_map.values())).keys())
        group = detect_point_group(first_band_ops)

        # Lookup frequencies and k-point details at Gamma (k=0) if freq_data is provided
        freq_lookup: Dict[int, float] = {}
        k_info: Dict[str, Any] = {"k_index": None, "k1": 0.0, "k2": 0.0, "k3": 0.0, "kmag_2pi": 0.0}

        pol_key = f"{parity}freqs"
        if freq_data and pol_key in freq_data:
            rows = freq_data[pol_key].get("rows", [])
            headers = freq_data[pol_key].get("headers", [])
            if rows and headers:
                # Check if kmag / kmag_2pi column is present in headers
                kmag_col = next((i for i, h in enumerate(headers) if "kmag" in h), None)
                if kmag_col is not None:
                    gamma_row = min(rows, key=lambda r: float(r[kmag_col]))
                    min_k_dist = float(gamma_row[kmag_col])
                else:
                    k1_idx = headers.index("k1") if "k1" in headers else 1
                    k2_idx = headers.index("k2") if "k2" in headers else 2
                    k3_idx = headers.index("k3") if "k3" in headers else 3
                    gamma_row = min(
                        rows,
                        key=lambda r: abs(float(r[k1_idx])) + abs(float(r[k2_idx])) + abs(float(r[k3_idx]))
                    )
                    min_k_dist = abs(float(gamma_row[k1_idx])) + abs(float(gamma_row[k2_idx])) + abs(float(gamma_row[k3_idx]))

                # Only assign frequencies and k-info if the closest k-point is strictly Gamma (|k| < 1e-3)
                if min_k_dist < 1e-3:
                    k1_idx = headers.index("k1") if "k1" in headers else 1
                    k2_idx = headers.index("k2") if "k2" in headers else 2
                    k3_idx = headers.index("k3") if "k3" in headers else 3
                    k_index_idx = headers.index("k_index") if "k_index" in headers else 0

                    if k_index_idx < len(gamma_row):
                        k_info["k_index"] = int(gamma_row[k_index_idx])
                    if k1_idx < len(gamma_row):
                        k_info["k1"] = float(gamma_row[k1_idx])
                    if k2_idx < len(gamma_row):
                        k_info["k2"] = float(gamma_row[k2_idx])
                    if k3_idx < len(gamma_row):
                        k_info["k3"] = float(gamma_row[k3_idx])
                    if kmag_col is not None and kmag_col < len(gamma_row):
                        k_info["kmag_2pi"] = float(gamma_row[kmag_col])

                    for col_idx, col_name in enumerate(headers):
                        if "band_" in col_name:
                            try:
                                b_num = int(col_name.split("band_")[1])
                                freq_lookup[b_num] = float(gamma_row[col_idx])
                            except (ValueError, IndexError):
                                pass

        for band_num, chars in sorted(band_map.items()):
            projections = compute_projections(chars, group=group)
            best_irrep, confidence = identify_irrep(projections)
            freq = freq_lookup.get(band_num, float("nan"))

            records.append({
                "parity": parity,
                "band": band_num,
                "freq": freq,
                "irrep": best_irrep,
                "confidence": round(confidence, 4),
                "point_group": group,
                "k_index": k_info.get("k_index"),
                "k1": k_info.get("k1", 0.0),
                "k2": k_info.get("k2", 0.0),
                "k3": k_info.get("k3", 0.0),
                "kmag_2pi": k_info.get("kmag_2pi", 0.0),
                "characters": {k: str(v) for k, v in chars.items()},
                "projections": {k: round(v, 4) for k, v in projections.items()},
            })

    return records
