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

# Simulation execution settings
simulation:
  ctl_script: "main.ctl"          # MPB control script file to execute (located in work_dir)
  work_dir: "."                   # Working directory containing ctl_script
  output_dir: "bo_output"         # Subfolder created inside work_dir for optimization output files
  cores: 4                        # Number of MPI cores per evaluation worker (single core if symmetries enabled)
  only_gamma: true                # Evaluate ONLY the Gamma point (k=0) during BO for 10x-50x speedup

# Optimization parameters (continuous search bounds: [min, max] in lattice units a)
parameters:
  r1: [0.15, 0.35]                # Primary cylinder radius r1
  r2: [0.15, 0.35]                # Secondary cylinder radius r2

# Fixed geometry & simulation parameters passed to MPB
fixed_parameters:
  resolution: 64                 # Grid resolution per unit cell length a
  num-bands: 12                  # Total bands calculated (keep >= 10 for high-order irrep doublets)
  h: 0.5                         # Slab thickness (in unit cell length a)

# Target band symmetry configuration
target:
  symmetry_group: "C4v"           # Point group symmetry: "C4v" (square lattice) or "C6v" (triangular lattice)
  polarization: "te"              # Polarization parity: "te" (Hz-odd/Ez-even) or "tm" (Ez-odd/Hz-even)
  
  # Option A: Dynamic Irreducible Representation (Irrep) Matching
  target_irreps: ["A_1", "E_1", "E_1"]  # Target irrep multiplet forming Dirac cone (e.g. A1 + E1)
  irrep_occurrences: [1, 1, 1]          # Occurrence index of each irrep above min_band (1 = 1st occurrence)
  min_band: 2                           # Lowest band index to inspect (excludes Band 1 acoustic mode)
  degeneracy_tol: 0.005                 # Frequency threshold (delta_omega) to trigger failsafe relabeling
  target_cost: 0.0025                   # Target cost floor below which active learning stops penalizing noise
  
  # Option B: Direct Static Mode Indices (Bypasses automatic irrep identification)
  bypass_irrep_identification: false    # Set to true to bypass irrep identification and track static bands
  mode_indices: [2, 3, 4]               # Explicit band indices to target directly when bypass is true

  # Option C: Slab Topology & Group Velocity
  check_slab_connectivity: true         # Invalidate disconnected dielectric geometries
  epsilon_threshold: 1.1                 # Dielectric threshold for matrix slab
  min_neck_width_px: 4                   # Reject connections narrower than 4 grid pixels wide
  compute_group_velocity: true           # Execute 2nd run at small delta_k to compute top target band group velocity
  delta_k: 0.01                          # Offset from Gamma point (k = (delta_k, 0, 0)) for group velocity

# Bayesian Optimizer & Gaussian Process (GP) settings
optimizer:
  grid_evaluation: false          # Set to true (or bypass_optimization: true) to evaluate uniform grid & fit GP
  grid_resolution: [10, 10]       # Uniform grid resolution per dimension (e.g. 10x10 = 100 points)
  
  max_iterations: 15              # Guided BO generations after initial sampling
  batch_size: 4                   # Candidates evaluated concurrently per generation
  initial_points: 16              # Total initial sampling points (e.g. 16 points = 4 initial batches of 4)
  initial_sampling: "sobol"       # Initial sampling method: "sobol" or "lhs" (Latin Hypercube)
  model: "GP"                     # Surrogate model: "GP" (Gaussian Process)
  acq_func: "LCB"                 # Acquisition function: "LCB" (Lower Confidence Bound) or "gp_hedge"
  acq_func_kwargs: {kappa: 3.5}   # Acquisition function kwargs (e.g. kappa=3.5 for LCB)
  objective_mode: "log"           # Cost metric mode: "log" (log10 cost) or "linear"
  strategy: "cl_min"              # Constant liar batch strategy: "cl_min", "cl_mean", or "cl_max"
  save_surrogate_freq: 1          # Save bo_surrogate_map.png every N generations (0 = only at end)
  random_state: 42                # Random seed for reproducibility
  neglect_sigma: false            # Set to true to neglect posterior uncertainty sigma in surrogate FOM map (plots 10^-mu)
  surrogate_colorbar_limits: [1, 1e3] # Colorbar limits [vmin, vmax] for surrogate map

# Postprocessing: Degeneracy Loci & Manifold Extraction
postprocessing:
  enabled: true                    # Set to true to extract continuous optimal 1D loci
  threshold_percentile: 90.0       # Percentile cutoff to isolate high-FOM candidate ridge (e.g. top 10%)
  min_locus_area_px: 25            # Minimum connected pixel area to qualify as a valid locus
  max_loci: 1                      # Maximum number of disjoint connected loci to extract and track
  smoothness: 0.001                # Spline smoothing regularization (s in scipy.interpolate.splprep)
  spline_degree: 3                 # Degree of B-spline (k=3 for cubic spline)
  sample_points: 50                # Number of evaluation points sampled along the locus curve
  export_csv: true                 # Export extracted curve coordinates to bo_locus.csv
  plot_overlay: true               # Overlay the extracted optimal curve on bo_surrogate_map.png
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
