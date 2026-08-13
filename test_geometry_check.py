"""
Unit Tests for Dielectric Geometry Connectivity Engine
"""

import numpy as np
import tempfile
import h5py
from pathlib import Path
from phc_nzi.geometry_check import check_array_connectivity, check_slab_connectivity

def test_continuous_slab():
    # 50x50 grid with dielectric matrix (eps=12) and central circular air hole (eps=1)
    grid_size = 50
    y, x = np.ogrid[:grid_size, :grid_size]
    center = grid_size / 2
    r = 15
    mask_hole = (x - center)**2 + (y - center)**2 <= r**2

    eps_grid = np.full((grid_size, grid_size), 12.0)
    eps_grid[mask_hole] = 1.0

    is_conn, num_comp, msg = check_array_connectivity(eps_grid, epsilon_threshold=1.1, check_pbc=True)
    assert is_conn is True
    print("Continuous slab test passed!")

def test_broken_slab():
    # 50x50 grid where two large overlapping air holes cut the dielectric matrix vertically
    grid_size = 50
    y, x = np.ogrid[:grid_size, :grid_size]

    # Large air channel cutting vertically from y=0 to y=50
    eps_grid = np.full((grid_size, grid_size), 12.0)
    eps_grid[:, 20:30] = 1.0  # Vertical air cut

    is_conn, num_comp, msg = check_array_connectivity(eps_grid, epsilon_threshold=1.1, check_pbc=True)
    assert is_conn is False
    assert "X-axis" in msg
    print("Broken slab test passed!")

def test_hdf5_connectivity_check():
    grid_size = 40
    eps_grid = np.full((grid_size, grid_size), 9.5)
    eps_grid[10:30, 10:30] = 1.0  # Central air block

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name

    try:
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("epsilon", data=eps_grid)

        is_conn, num_comp, msg = check_slab_connectivity(h5_path, epsilon_threshold=1.1)
        assert is_conn is True
        print("HDF5 connectivity check test passed!")
    finally:
        p = Path(h5_path)
        if p.exists():
            p.unlink()

def test_narrow_neck_rejection():
    # 50x50 grid where left and right blocks span Y (0:50) and are connected ONLY by a 2-pixel wide horizontal bridge
    grid_size = 50
    eps_grid = np.full((grid_size, grid_size), 1.0)
    eps_grid[0:50, 0:20] = 12.0
    eps_grid[0:50, 30:50] = 12.0
    eps_grid[24:26, 20:30] = 12.0  # 2-pixel wide neck connecting left and right blocks

    # With min_neck_width_px=1, it is connected
    is_conn1, _, _ = check_array_connectivity(eps_grid, epsilon_threshold=1.1, check_pbc=True, min_neck_width_px=1)
    assert is_conn1 is True

    # With min_neck_width_px=4, the 2-pixel wide bridge is severed by morphological opening
    is_conn4, _, msg4 = check_array_connectivity(eps_grid, epsilon_threshold=1.1, check_pbc=True, min_neck_width_px=4)
    assert is_conn4 is False
    assert "too narrow" in msg4
    print("Narrow neck rejection test passed!")

if __name__ == "__main__":
    test_continuous_slab()
    test_broken_slab()
    test_hdf5_connectivity_check()
    test_narrow_neck_rejection()
    print("All geometry connectivity unit tests passed successfully!")
