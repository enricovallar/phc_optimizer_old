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

## Bayesian Optimization (`phc-bo`) for Dirac Cone Search

Find geometric parameters that minimize the frequency gap between target symmetry representations (e.g., $A_2 + E$) at the $\Gamma$ point using Gaussian Process surrogate modeling.

### CLI Usage

```bash
# Run Bayesian Optimization with default config (ctl/bo_config.yaml):
uv run phc-bo --config ctl/bo_config.yaml --dir work
```

### Optimization Configuration (`bo_config.yaml`)

```yaml
# Simulation environment and execution settings
simulation:
  ctl_script: "main.ctl"
  work_dir: "work"
  output_dir: "bo_output"
  cores: 4
  parallel_workers: 4
  only_gamma: true                 # Evaluate ONLY Gamma point during BO iterations (10x-50x speedup!)
  debug_timing: true

parameters:
  r1: [0.15, 0.35]                 # Continuous bounds for r1
  r2: [0.15, 0.35]                 # Continuous bounds for r2

fixed_parameters:
  resolution: 64
  num-bands: 12
  h: 0.5

target:
  symmetry:
    group: "C4v"                  # Point group ("C4v" or "C6v")
    polarization: "te"             # Polarization ("te" or "tm")
    target_irreps: ["A_2", "E", "E"] # Irrep multiplet to form Dirac cone
    irrep_occurrences: [1, 1, 1]
    min_band: 2
    degeneracy_tol: 0.005          # Degeneracy failsafe threshold for mode mixing
    target_cost: 0.0025            # Active learning cost threshold floor
    bypass_irrep_identification: false
    mode_indices: [2, 3, 4]       # Explicit band indices (when bypassing irrep detection)

  connectivity:
    enabled: true                 # Invalidate disconnected matrix slab geometries
    epsilon_threshold: 1.1        # Dielectric threshold to identify matrix slab
    min_neck_width_px: 4          # Reject connections narrower than 4 grid pixels wide

  group_velocity:
    enabled: true                 # Compute group velocity during BO iterations
    delta_k: 0.01                 # Small k-vector offset from Gamma for group velocity
    optimal_only: true            # Only compute vg for purple/low-cost optimal points

optimizer:
  iterations:
    max_iterations: 15
    batch_size: 4
    initial_points: 16
    initial_sampling: "sobol"

  surrogate:
    model: "GP"
    objective_mode: "log"         # "log" (log10 cost) or "linear"
    strategy: "cl_min"
    random_state: 42

  acquisition:
    acq_func: "LCB"               # "LCB" (Lower Confidence Bound) or "gp_hedge"
    acq_func_kwargs: {kappa: 3.5}
    optimizer: "sampling"
    n_points: 500
    n_restarts: 1

  visualization:
    save_surrogate_freq: 1
    neglect_sigma: false
    colorbar_limits: [1, 1e3]

postprocessing:
  enabled: true                    # Enable postprocessing pipeline

  locus:
    threshold_percentile: 70.0    # Percentile cutoff to isolate the continuous high-FOM ridge
    min_locus_area_px: 25         # Minimum connected pixel area to qualify as a valid locus
    max_loci: 1                   # Maximum number of disjoint loci to extract
    smoothness: 0.001             # Spline smoothing regularization (s in scipy.interpolate.splprep)
    spline_degree: 3              # Degree of B-spline (k=3 for cubic spline)
    sample_points: 20             # Number of evaluation points sampled along the locus curve

  refinement:
    enabled: true                 # Fine-tune sampled points to exact degeneracy (residual gap < tolerance)
    tolerance: 1.0e-5             # Target residual gap tolerance floor (|Δω/ω0| < 1e-5)
    max_steps: 6                  # Maximum Γ-only root-finding evaluations per point (typically 2-3)
    method: "normal"              # "normal" (orthogonal to curve) or "r2" / "r1"

  group_velocity:
    enabled: true                 # Calculate group velocity at each refined locus point in parallel
    delta_k: 0.01

  output:
    export_csv: true              # Export extracted curve coordinates to bo_locus.csv
    plot_overlay: true            # Overlay both GP surrogate and refined loci on bo_surrogate_map.png
    plot_profiles: true           # Generate 3-panel bo_locus_profile.png
```

### Python API Usage

```python
from phc_nzi import run_bo, BayesianOptimizer

# Run optimization using YAML configuration
results = run_bo(config_path="InP/bo_config.yaml", work_dir="InP")

print("Optimal parameters:", results["optimal_parameters"])
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



