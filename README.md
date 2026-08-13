# phc-nzi

Modular MPB configuration, HPC runner, and frequency data extractor package for photonic crystal simulations on DTU DCC.

## Setup with `uv`

Install `uv` (if not already installed) and sync the project virtual environment:

```bash
uv sync
```

## Running Simulations

You can run MPB simulations using `uv`:

```bash
# Run with default options:
uv run phc-runner

# Specify custom target working directory, script, cores, and MPB parameters:
uv run phc-runner --dir work --script main.ctl --cores 8 -p h=0.6 r1=0.25 r2=0.15 resolution=64
```

### CLI Arguments (`phc-runner --help`)

| Option | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--dir` | `-d` | Target working directory for execution & logs | Current Working Directory |
| `--script` | `-s` | Control file (`.ctl`) to run | `example.ctl` |
| `--cores` | `-c` | Number of MPI cores to use | `4` |
| `--param` | `-p` | Pass key=value parameters to MPB (e.g. `-p h=0.6 r1=0.25 resolution=64`) | None |
| `--no-mpi` | | Disable MPI parallel execution | False |
| `--symmetry` | | Enable Gamma point symmetry display and irrep classification | False |
| `--group-velocity` | `--vg` | Enable group velocity calculation for all k-points and bands | False |

---

## Simulation Output Directory Structure (`output/`)

All generated runtime output files are automatically isolated inside an **`output/`** subfolder within your working directory (e.g. `work/output/`), keeping your control scripts clean and making output cleanup trivial (`rm -rf work/output/`).

```text
v2/work/                          <-- Working Directory (Clean!)
├── main.ctl                      <-- Input Control Script (Untouched)
└── output/                       <-- Single Folder for ALL Generated Outputs
    ├── output.out                <-- MPB simulation log
    ├── error.err                 <-- Execution error log
    ├── main-epsilon.h5           <-- Raw dielectric HDF5 grid
    ├── main-epsilon.converted.h5 <-- Rectified Cartesian dielectric HDF5 grid
    ├── tefreqs.data              <-- TE band frequencies (matrix format)
    ├── tmfreqs.data              <-- TM band frequencies (matrix format)
    ├── kpath_labels.data         <-- High-symmetry k-path labels
    ├── symmetries.json           <-- Irrep classifications & character tables at Gamma
    ├── tevelocity.data           <-- TE group velocities (matrix format)
    ├── tmvelocity.data           <-- TM group velocities (matrix format)
    ├── group_velocities.json     <-- Group velocity vectors & magnitudes (JSON format)
    ├── band_structure.png        <-- Band structure plot figure
    └── epsilon_map.png           <-- Dielectric map plot figure
```

---

## Extracting Frequency Data (`.data` files)

The package automatically parses simulation logs (`output/output.out`) into clean `.data` and `.json` files.

You can also run the extractor manually on any MPB log file:

```bash
uv run phc-extractor -i work/output/output.out -o work/output/
```

### Output `.data` File Format (`tefreqs.data`)

```text
# k_index k1 k2 k3 kmag_2pi te_band_1 te_band_2 te_band_3 te_band_4 te_band_5
1 -0.05 0.05 0 0.1 0.0333558 0.352495 0.357269 0.381788 0.385973
2 -0.0375 0.0375 0 0.075 0.0250171 0.359292 0.363945 0.381189 0.385331
...
```

---

## Python API Usage

```python
from phc_nzi import run_hpc, extract_frequencies

# 1. Run MPB simulation with custom parameters
run_hpc(
    script="main.ctl",
    mpb_command_line_params={
        "h": 0.6,
        "r1": 0.25,
        "r2": 0.15,
        "resolution": 64,
        "num-bands": 12,
    },
    use_mpi=True,
    cores=4,
    wd="work"
)

# 2. Extract frequency data and symmetry classifications manually if needed
data = extract_frequencies(output_path="work/output.out")

# 3. Read extracted symmetries
from phc_nzi import load_symmetries

symmetries = load_symmetries("work/symmetries.data")
# Or inspect from in-memory dictionary:
sym_data = data.get("symmetries")
```

---

## Symmetry & Irreducible Representation (Irrep) Classification at $\Gamma$

When `display-symmetry?` / `display_symmetry?` is enabled, the runner and extractor automatically analyze point-group symmetries ($C_{4v}$ / $C_{6v}$) at the $\Gamma$ point ($k=0$) and project band states onto irreducible representations ($A_1, A_2, B_1, B_2, E, E_1, E_2$).

### Output Files Generated:
* **`symmetries.data`**: Space-delimited table of assigned irreps and confidence scores.
  ```text
  # parity band freq irrep confidence point_group
  te 1 0.24513 A_1 1.0000 C4v
  te 2 0.38912 E 1.0000 C4v
  te 3 0.38912 E 1.0000 C4v
  ```
* **`symmetries.json`**: Structured JSON containing raw character expectation values ($C_4, C_6, C_3, C_2, \sigma_v, \sigma_d$) and full projection breakdowns.

---

## Group Velocity Extraction ($\vec{v}_g$)

When `--group-velocity` (or `display-group-velocity?=true`) is enabled, the runner and extractor calculate and parse group velocity vector components $(v_x, v_y, v_z)$ and magnitude $|\vec{v}_g| = \sqrt{v_x^2 + v_y^2 + v_z^2}$ for all bands and k-points along the reciprocal space trajectory.

### Output Files Generated:
* **`group_velocities.data`**: Flat table containing all bands, k-points, vector components, and speeds:
  ```text
  # parity band k_index k1 k2 k3 kmag_2pi vx vy vz vg_mag
  te 1 1 -0.05 0.05 0 0.1 0.012 0.015 0 0.01921458
  ```
* **`tevelocity.data` / `tmvelocity.data`**: Matrix format with band-by-band components and speeds matching the shape of `tefreqs.data`.
* **`group_velocities.json`**: Structured JSON file for programmatic reading.


