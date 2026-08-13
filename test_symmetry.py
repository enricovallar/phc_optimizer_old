import sys
from pathlib import Path

# Ensure v2/src is in sys.path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from phc_nzi.symmetry import (
    detect_point_group,
    compute_projections,
    identify_irrep,
    analyze_symmetries_from_log,
)
from phc_nzi.extractor import extract_symmetries, load_symmetries


def test_symmetry_logic():
    print("--- Test 1: Point Group Detection ---")
    assert detect_point_group(["C4", "C2", "sv", "sd"]) == "C4v"
    assert detect_point_group(["C6", "C3", "C2", "sv", "sd"]) == "C6v"
    print("Point group detection passed!")

    print("\n--- Test 2: Projections for C4v ---")
    # A1 representation in C4v: C4=1, C2=1, sv=1, sd=1
    chars_a1 = {"C4": complex(1.0), "C2": complex(1.0), "sv": complex(1.0), "sd": complex(1.0)}
    projs_a1 = compute_projections(chars_a1, group="C4v")
    irrep_a1, conf_a1 = identify_irrep(projs_a1)
    assert irrep_a1 == "A_1"
    assert abs(conf_a1 - 1.0) < 1e-4
    print(f"C4v A_1 test passed! Identified: {irrep_a1} ({conf_a1:.4f})")

    # B1 representation in C4v: C4=-1, C2=1, sv=1, sd=-1
    chars_b1 = {"C4": complex(-1.0), "C2": complex(1.0), "sv": complex(1.0), "sd": complex(-1.0)}
    projs_b1 = compute_projections(chars_b1, group="C4v")
    irrep_b1, conf_b1 = identify_irrep(projs_b1)
    assert irrep_b1 == "B_1"
    assert abs(conf_b1 - 1.0) < 1e-4
    print(f"C4v B_1 test passed! Identified: {irrep_b1} ({conf_b1:.4f})")

    # E representation in C4v (single state): C4=0, C2=-1, sv=0, sd=0
    chars_e = {"C4": complex(0.0), "C2": complex(-1.0), "sv": complex(0.0), "sd": complex(0.0)}
    projs_e = compute_projections(chars_e, group="C4v")
    irrep_e, conf_e = identify_irrep(projs_e)
    assert irrep_e == "E"
    assert abs(conf_e - 1.0) < 1e-4
    print(f"C4v E test passed! Identified: {irrep_e} ({conf_e:.4f})")

    print("\n--- Test 3: Log Parsing & Analysis ---")
    mock_log = """
    Initializing MPB...
    SYM_DATA_START_te
    te, 1, C4=1.0+0.0i, C2=1.0+0.0i, sv=1.0+0.0i, sd=1.0+0.0i
    te, 2, C4=0.0+0.0i, C2=-1.0+0.0i, sv=0.0+0.0i, sd=0.0+0.0i
    te, 3, C4=0.0+0.0i, C2=-1.0+0.0i, sv=0.0+0.0i, sd=0.0+0.0i
    te, 4, C4=-1.0+0.0i, C2=1.0+0.0i, sv=1.0+0.0i, sd=-1.0+0.0i
    SYM_DATA_END_te
    """

    records = analyze_symmetries_from_log(mock_log)
    assert len(records) == 4
    assert records[0]["irrep"] == "A_1"
    assert records[1]["irrep"] == "E"
    assert records[2]["irrep"] == "E"
    assert records[3]["irrep"] == "B_1"
    print("Mock log parsing passed successfully!")
    print(records)


if __name__ == "__main__":
    test_symmetry_logic()
