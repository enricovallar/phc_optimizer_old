import os
import yaml
import tempfile
from phc_nzi.bo import load_bo_config, find_bands_from_irreps, failsafe_irrep_mapping

def test_bo_config_loading():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump({
            "simulation": {"ctl_script": "test.ctl", "only_gamma": True},
            "parameters": {"r1": [0.1, 0.4], "r2": [0.2, 0.5]},
            "target": {"target_irreps": ["A_2", "E", "E"]},
            "optimizer": {"max_iterations": 10}
        }, f)
        temp_name = f.name

    try:
        config = load_bo_config(temp_name)
        assert config["simulation"]["ctl_script"] == "test.ctl"
        assert config["simulation"]["only_gamma"] is True
        assert config["parameters"]["r1"] == [0.1, 0.4]
        assert "h" not in config["parameters"]  # Verify h is omitted per user request
        assert config["target"]["target_irreps"] == ["A_2", "E", "E"]
        assert config["optimizer"]["max_iterations"] == 10
        print("BO config loading test passed!")
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

def test_failsafe_irrep_mapping():
    # Construct scrambled full_irrep_map where band 3 is mislabeled as 'B_2' due to eigenmode mixing
    # Target irreps: ['A_2', 'E', 'E']
    full_irrep_map = {
        2: ("E", 0.5, 0.3501),
        3: ("B_2", 0.5, 0.3502),  # Mislabeled state due to mixing
        4: ("E", 0.5, 0.3503)
    }
    bands_to_check = [2, 3, 4]
    failsafe_irrep_mapping(
        target_irreps=["A_2", "E", "E"],
        degeneracy_tol=0.001,
        full_irrep_map=full_irrep_map,
        bands_to_check=bands_to_check
    )

    # Verify relabeling to target multiplet ['A_2', 'E', 'E']
    assert full_irrep_map[2][0] == "A_2"
    assert full_irrep_map[3][0] == "E"
    assert full_irrep_map[4][0] == "E"
    print("Failsafe irrep mapping test passed!")

def test_grid_generation():
    from phc_nzi.bo import BayesianOptimizer
    config = {
        "simulation": {"ctl_script": "test.ctl"},
        "parameters": {"r1": [0.1, 0.4], "r2": [0.2, 0.5]},
        "target": {},
        "optimizer": {"grid_evaluation": True, "grid_resolution": [3, 4]}
    }
    opt = BayesianOptimizer(config)
    pts = opt._generate_grid_points()
    assert len(pts) == 12  # 3 x 4 grid
    assert pts[0] == [0.1, 0.2]
    assert pts[-1] == [0.4, 0.5]
    print("Grid generation test passed!")

def test_bypass_irrep():
    from phc_nzi.bo import BayesianOptimizer
    config = {
        "simulation": {"ctl_script": "test.ctl"},
        "parameters": {"r1": [0.1, 0.4]},
        "target": {"bypass_irrep_identification": True, "mode_indices": [2, 3, 4]},
        "optimizer": {}
    }
    opt = BayesianOptimizer(config)
    assert opt.target_cfg.get("mode_indices") == [2, 3, 4]
    assert opt.target_cfg.get("bypass_irrep_identification") is True
    print("Bypass irrep identification test passed!")

if __name__ == "__main__":
    test_bo_config_loading()
    test_failsafe_irrep_mapping()
    test_grid_generation()
    test_bypass_irrep()
    print("All BO unit tests passed successfully!")
