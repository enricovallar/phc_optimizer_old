# phc-nzi

Modular MPB configuration, HPC runner, and frequency data extractor package for photonic crystal simulations on DTU DCC.

## Setup with `uv`

Install `uv` (if not already installed) and sync the project virtual environment:

```bash
uv sync
```

## Initializing a Simulation Directory (`phc-init`)

Initialize a directory populated with fully documented `bo_config.yaml` and `main.ctl` template files:

```bash
# Initialize a new project directory (e.g. my_simulation):
uv run phc-init my_simulation
```

---

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

---

## Bayesian Optimization (`phc-bo`) for Dirac Cone Search (Hydra Powered)

Find geometric parameters that minimize the frequency gap between target symmetry representations (e.g., $A_2 + E$) at the $\Gamma$ point using Gaussian Process surrogate modeling and Hydra modular configuration.

### Initializing a Project Workspace (`phc-init`)

```bash
uv run phc-init my_simulation
```

This creates the modular project structure:
```text
my_simulation/
├── main.ctl                         # MPB Scheme script (materials, lattice, geometry)
└── configs/                         # Modular Hydra configuration
    ├── config.yaml                  # Master composition entrypoint
    ├── simulation/default.yaml      # HPC workers, cores, only_gamma, ctl_script
    ├── parameters/default.yaml      # Search bounds (r1, r2) & fixed MPB parameters (h, res, sz)
    ├── target/accidental_dirac.yaml # Bands, symmetry, polarization, neck connectivity
    ├── optimizer/bayesian.yaml      # Batch size, surrogate GP, LCB acquisition (kappa)
    └── postprocessing/default.yaml  # Locus extraction, refinement, vg, band diagram
```

### CLI Usage & Hydra Overrides

```bash
# Run with defaults:
uv run phc-bo

# Override parameters dynamically from CLI:
uv run phc-bo parameters.fixed.h=0.35 simulation.parallel_workers=28

# Multi-thickness parameter sweeps (-m / --multirun):
uv run phc-bo -m parameters.fixed.h=0.30,0.35,0.40,0.45,0.50

# Fast postprocessing on existing simulation output:
uv run phc-bo general.postprocess_only=true postprocessing.band_diagram.enabled=false
```

# 2. Re-plot figures from previous data without running any simulations:
uv run phc-bo general.plot_only=true
```

### Python API Usage

```python
from phc_nzi import BayesianOptimizer, load_bo_config

# Run optimization programmatically
opt = BayesianOptimizer(config_dict)
results = opt.run()

# Or run postprocessing only:
opt.postprocess_only()
```

### Generated Optimization Output Structure (`bo_output/`):

```text
<bo_output>/
├── bo_convergence.png         # Optimization convergence & sample trajectory plot
├── bo_surrogate_map.png       # 2-panel surrogate FOM map with green dashed optimal locus overlay
├── bo_loci.json               # JSON metadata and coordinates for all extracted loci
├── bo_model.pkl               # Serialized GP surrogate model
│
├── locus_01/                  # Subfolder for Locus #1
│   ├── bo_locus_profile.png   # 3-panel profile: (a) (r1, r2) Trajectory with Vg colormap, (b) Vg(t), (c) FOM(t)
│   ├── bo_locus.csv           # Coordinates (r1, r2), FOM, residual gap, and group velocity
│   │
│   ├── pt_01/                 # Point #1
│   │   ├── point_info.json    # Parameter coordinates, residual gap, and delta_k metrics
│   │   ├── band_structure/    # Full k-path MPB simulation & plots
│   │   │   ├── band_structure.png
│   │   │   ├── epsilon_map.png
│   │   │   ├── output.out
│   │   │   └── symmetries.json
│   │   └── group_velocity/    # Dedicated MPB simulation at k = (delta_k, 0, 0)
│   │       ├── output.out
│   │       ├── group_velocities.json
│   │       └── tevelocity.data
│   │
│   ├── pt_02/
│   │   └── ...
│   └── pt_N/
│       └── ...
│
└── locus_02/                  # (If multiple loci are detected)
    ├── bo_locus_profile.png
    ├── bo_locus.csv
    ├── pt_01/
    └── ...
```

---

## 3D Slab Wavelength-Tuning & Universal Design Curves (`phc-slab-sweep` & `phc-design`)

Based on the physical scaling principles in *Vallar et al., Optical Materials Express 16, 2681–2695 (2026)*:
1. **Master Dimensionless Sweep (`phc-slab-sweep`)**: Sweeps normalized slab thickness ratios $h/a$ (e.g. $[0.30, 0.35, 0.40, 0.45, 0.50]$), running Bayesian Optimization and extracting optimal Dirac cone loci for each $h/a$.
2. **Universal Design Curves (`phc-design`)**: Ingests the master sweep dataset and applies scale-invariance scaling laws to generate continuous design curves ($a(h)$, $r_1(h)$, $r_2(h)$, $FF(h)$, $v_g(h)$) for **any** target operational wavelength $\lambda_0$ (e.g. $1550\text{ nm}$).

### CLI Usage

```bash
# 1. Run multi-thickness sweep over normalized slab thicknesses h/a:
uv run phc-slab-sweep --config "InP slab/bo_config.yaml" --h-ratios 0.30 0.35 0.40 0.45 0.50

# 2. Generate publication-quality design curves for a target wavelength:
uv run phc-design --input "InP slab/sweep_results" --wavelength 1550 --output "InP slab/design_rules_1550nm.png" --highlight 375.0
```

### Python API Usage

```python
from phc_nzi import run_slab_sweep, generate_design_curves, SlabDesignCurves

# Run sweep across slab thicknesses
sweep_res = run_slab_sweep(
    config_path="InP slab/bo_config.yaml",
    h_ratios=[0.30, 0.35, 0.40, 0.45, 0.50],
    target_wavelength_nm=1550.0,
)

# Query design curves for a specific physical slab thickness (e.g. 375 nm):
curves = SlabDesignCurves("InP slab/sweep_results/slab_sweep_summary.csv", lambda0_nm=1550.0)
design = curves.query(h_nm=375.0)

print(f"Target h = 375 nm @ 1550 nm:")
print(f"  Lattice constant a: {design['a_nm']:.1f} nm")
print(f"  Hole radius r1:     {design['r1_nm']:.1f} nm")
print(f"  Hole radius r2:     {design['r2_nm']:.1f} nm")
print(f"  Filling factor FF:  {design['filling_factor']:.3f}")
print(f"  Group velocity:     {design['vg_over_c']:.3f} c")
```
