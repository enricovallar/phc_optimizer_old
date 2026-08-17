"""
Project Directory Initializer for Photonic Crystal MPB Simulations & Bayesian Optimization
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Union, Tuple

BO_CONFIG_TEMPLATE = """# ==============================================================================
# Photonic Crystal NZI & Dirac-like Cone Bayesian Optimization Configuration
# ==============================================================================

# Simulation environment and HPC execution settings
simulation:
  ctl_script: "main.ctl"          # MPB control script file to execute (in work_dir)
  work_dir: "."                   # Working directory containing ctl_script
  output_dir: "bo_output"         # Subfolder created inside work_dir for outputs
  cores: 4                        # Number of MPI cores per evaluation worker
  parallel_workers: 4             # Number of concurrent parallel simulation workers
  only_gamma: true                # Evaluate ONLY the Gamma point (k=0) during BO for speedup
  debug_timing: true              # Detailed timing breakdown logs & reports

# Optimization parameters (continuous search bounds: [min, max] in lattice units a)
parameters:
  r1: [0.15, 0.35]                # Primary cylinder radius r1
  r2: [0.15, 0.35]                # Secondary cylinder radius r2

# Fixed geometry and simulation parameters passed to MPB
fixed_parameters:
  resolution: 64                 # Grid resolution per unit cell length a
  num-bands: 12                  # Total bands calculated (keep >= 10 for high-order irreps)
  h: 0.5                         # Slab thickness (in unit cell length a)

# Target physical dispersion & symmetry specifications
target:
  symmetry:
    group: "C4v"                  # Point group symmetry: "C4v" (square) or "C6v" (triangular)
    polarization: "te"             # Polarization parity: "te" (Hz-odd/Ez-even) or "tm" (Ez-odd/Hz-even)
    target_irreps: ["A_1", "E_1", "E_1"] # Target irrep multiplet forming Dirac cone (e.g. A1 + E1)
    irrep_occurrences: [1, 1, 1]  # Target occurrence order of irreps
    min_band: 2                   # Lowest band index to inspect (excludes Band 1 acoustic mode)
    degeneracy_tol: 0.005         # Frequency threshold (delta_omega) to trigger failsafe relabeling
    target_cost: 0.0025           # Active learning target cost floor
    bypass_irrep_identification: false # Set true to target mode_indices directly
    mode_indices: [2, 3, 4]       # Explicit band indices (when bypassing irrep detection)

  connectivity:
    enabled: true                 # Invalidate disconnected matrix slab geometries
    epsilon_threshold: 1.1        # Permittivity threshold for matrix slab
    min_neck_width_px: 4          # Reject connections narrower than 4 grid pixels wide

  group_velocity:
    enabled: true                 # Compute group velocity during BO iterations
    delta_k: 0.01                 # Offset from Gamma for group velocity calculation
    optimal_only: true            # Only compute vg for purple/low-cost optimal points

# Bayesian Optimizer & Surrogate Model Settings
optimizer:
  iterations:
    max_iterations: 15            # Guided BO generations after initial sampling
    batch_size: 4                 # Candidates evaluated concurrently per generation
    initial_points: 16            # Total initial sampling points
    initial_sampling: "sobol"     # Initial sampling method: "sobol" or "lhs"

  surrogate:
    model: "GP"                   # Surrogate model: "GP", "RF", "ET", "GBRT"
    objective_mode: "log"         # Cost metric mode: "log" (log10 cost) or "linear"
    strategy: "cl_min"            # Constant liar batch strategy: "cl_min", "cl_mean", "cl_max"
    random_state: 42              # Random seed for reproducibility

  acquisition:
    acq_func: "LCB"               # Acquisition function ("LCB", "EI", "PI", "gp_hedge")
    acq_func_kwargs:
      kappa: 3.5                  # Exploration parameter kappa for LCB
    optimizer: "sampling"         # Acquisition optimizer ("sampling" or "lbfgs")
    n_points: 500                 # Sampling candidates for acquisition maximization
    n_restarts: 1                 # Restarts for acquisition optimizer

  grid:
    enabled: false                # Bypass BO and evaluate uniform parameter grid
    resolution: [10, 10]          # Uniform grid resolution per dimension

  visualization:
    save_surrogate_freq: 1        # Save bo_surrogate_map.png every N generations (0 = only at end)
    neglect_sigma: false          # Neglect posterior uncertainty sigma in surrogate map
    colorbar_limits: [1, 1e3]     # FOM colorbar scale limits

# Postprocessing: Degeneracy Loci Extraction & Analysis Pipeline
postprocessing:
  enabled: true                   # Enable postprocessing pipeline

  # 1. Manifold Extraction & Skeletonization
  locus:
    threshold_percentile: 70.0    # Percentile cutoff to isolate high-FOM ridge (e.g. top 30%)
    min_locus_area_px: 25         # Minimum connected component size in pixels
    max_loci: 1                   # Maximum number of disjoint loci to extract
    smoothness: 0.001             # B-spline smoothing regularization factor
    spline_degree: 3              # Degree of B-spline (k=3 for cubic spline)
    sample_points: 20             # Number of evaluation points sampled along the curve

  # 2. Degeneracy Fine-Tuning (Gamma-only normal line search)
  refinement:
    enabled: true                 # Fine-tune sampled points to exact Gamma degeneracy
    tolerance: 1.0e-5             # Target residual gap floor (|Δω/ω0| < 1e-5)
    max_steps: 6                  # Max Gamma-only line search evaluations per point
    method: "normal"              # "normal" (orthogonal to curve) or "r2" / "r1"

  # 3. Group Velocity Evaluation along Loci
  group_velocity:
    enabled: true                 # Compute target-band group velocities along the locus in parallel
    delta_k: 0.01                 # Offset from Gamma for vg calculation

  # 4. Output & Visualization
  output:
    export_csv: true              # Export bo_locus.csv inside each locus folder
    plot_overlay: true            # Overlay GP surrogate and refined loci on bo_surrogate_map.png
    plot_profiles: true           # Generate 3-panel bo_locus_profile.png
"""

MAIN_CTL_TEMPLATE = """; ==============================================================================
; MPB Control File: Modular Photonic Crystal Simulation
; ==============================================================================

(load-module "materials.ctl")
(load-module "parity_functions.ctl")
(load-module "shapes.ctl")
(load-module "wyckoff.ctl")
(load-module "custom_nonbloch_output.ctl")
(load-module "lattices.ctl")

; Geometric parameters (overridden by command line / BO runner)
(define-param h 0.5)
(define-param r1 0.25)
(define-param r2 0.15)
(define-param resolution 64)
(define-param num-bands 12)

; Material definitions
(define matrix-mat (make dielectric (epsilon 12.0)))
(define atom-mat (make dielectric (epsilon 1.0)))

; Lattice geometry (C4v square lattice or C6v triangular lattice)
(set! geometry-lattice (make-square-lattice no-size))

; Background slab and parametric Wyckoff cylinder shapes
(define background-slab
  (make block (size (vector3 1e20 1e20 h))
              (center (vector3 0 0 0))
              (material matrix-mat)))

(define shapes-1
  (map (lambda (pos) (make-param-cylinder pos h r1 atom-mat))
       (get-C4v-1a)))

(define shapes-2
  (map (lambda (pos) (make-param-cylinder pos h r2 atom-mat))
       (get-C4v-1b)))

(set! geometry (make-superposition background-slab (list shapes-1 shapes-2)))

; High-symmetry k-path trajectory
(define-param kmag 0.1)
(set! k-points (interpolate 10 (get-sq-path-circular kmag)))
(display-kpath-labels sq-labels-circular)

; Override k-points to Gamma point (0 0 0) if only-gamma? flag is set during BO
(if (or only-gamma? only_gamma?)
    (set! k-points (list (vector3 0 0 0))))

; Solver callbacks
(if run-te? (run-solver-with-callbacks run-te))
(if run-tm? (run-solver-with-callbacks run-tm))
(if run-zeven? (run-solver-with-callbacks run-zeven))
(if run-zodd? (run-solver-with-callbacks run-zodd))
"""

def init_folder(target_dir: Union[str, Path] = ".") -> Tuple[Path, Path]:
    """
    Initializes a target directory with fully documented bo_config.yaml and main.ctl template files.
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    yaml_file = target_path / "bo_config.yaml"
    ctl_file = target_path / "main.ctl"

    if not yaml_file.exists():
        yaml_file.write_text(BO_CONFIG_TEMPLATE)
        print(f"Created configuration file: '{yaml_file}'")
    else:
        print(f"Configuration file already exists: '{yaml_file}'")

    if not ctl_file.exists():
        ctl_file.write_text(MAIN_CTL_TEMPLATE)
        print(f"Created MPB control script: '{ctl_file}'")
    else:
        print(f"MPB control script already exists: '{ctl_file}'")

    return yaml_file, ctl_file

def main():
    parser = argparse.ArgumentParser(
        description="Initialize a directory with fully documented bo_config.yaml and main.ctl templates."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Target directory to initialize (default: current directory)"
    )
    parser.add_argument(
        "--dir", "-d",
        dest="dir_opt",
        default=None,
        help="Target directory option"
    )

    args = parser.parse_args()
    target_dir = args.dir_opt or args.directory

    print("==================================================================")
    print("Initializing Photonic Crystal Simulation Directory")
    print(f"Target Directory: '{target_dir}'")
    print("==================================================================")

    yaml_file, ctl_file = init_folder(target_dir)

    print("\nInitialization Complete!")
    print("You can now edit the generated config and control files, then run:")
    print(f"  uv run phc-runner --dir {target_dir}")
    print(f"  uv run phc-bo --config {yaml_file} --dir {target_dir}")
    print("==================================================================")

if __name__ == "__main__":
    main()
