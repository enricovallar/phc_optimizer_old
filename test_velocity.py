import sys
from pathlib import Path

# Ensure v2/src is in sys.path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from phc_nzi.extractor import extract_group_velocities, load_group_velocities


def test_velocity_extraction(tmp_path):
    mock_log = """
    Initializing MPB...
    tevelocity:, 1, #(0.01 0.02 0.0), #(0.03 -0.04 0.0)
    tevelocity:, 2, #(0.05 0.00 0.0), #(0.00 0.12 0.0)
    """

    log_file = tmp_path / "output.out"
    log_file.write_text(mock_log)

    res = extract_group_velocities(output_path=log_file, output_dir=tmp_path, save_data=True)

    assert "tevelocity" in res
    assert len(res["flat_records"]) == 4

    # Test band 2 at k_idx 1 (vx=0.03, vy=-0.04, vz=0.0) -> vg_mag = 0.05
    rec_b2 = [r for r in res["flat_records"] if r["k_index"] == 1 and r["band"] == 2][0]
    assert abs(rec_b2["vx"] - 0.03) < 1e-5
    assert abs(rec_b2["vy"] - (-0.04)) < 1e-5
    assert abs(rec_b2["vg_mag"] - 0.05) < 1e-5
    print("Group velocity parsing & magnitude calculation passed!")

    # Test disk loading
    loaded = load_group_velocities(tmp_path / "group_velocities.json")
    assert len(loaded) == 4
    assert loaded[0]["parity"] == "te"
    print("Disk loading test passed!")


if __name__ == "__main__":
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_velocity_extraction(Path(tmpdir))
