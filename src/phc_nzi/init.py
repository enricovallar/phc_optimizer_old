"""
Project Directory Initializer for Photonic Crystal MPB Simulations & Bayesian Optimization
"""

import shutil
import argparse
from pathlib import Path
from typing import Union, Tuple

MAIN_CTL_TEMPLATE = """; ==============================================================================
; MPB Control File: Modular 3D Photonic Crystal Slab Membrane Simulation
; ==============================================================================

(load-module "materials.ctl")
(load-module "parity_functions.ctl")
(load-module "shapes.ctl")
(load-module "wyckoff.ctl")
(load-module "custom_nonbloch_output.ctl")
(load-module "lattices.ctl")

; Geometric and computational parameters (passed via CLI from BO optimizer)
(define-param h 0.5)
(define-param r1 0.25)
(define-param r2 0.25)
(define-param sz 4)
(define-param resolution 25)
(define-param res-z 16)
(define-param num-bands 12)

; Material definitions (InP dielectric slab in air)
(define InP (make dielectric (epsilon 10.0489)))
(define matrix-mat InP)
(define atom-mat air)

; Lattice geometry: C4v square lattice with supercell height sz
(set! geometry-lattice (make-square-lattice sz))

; Slab geometry and Wyckoff cylinder air-holes
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

(define-param delta-k 0.001)
; Override k-points for small delta-k group velocity evaluation
(if (or delta-k-mode? delta_k_mode?)
    (set! k-points (list (vector3 delta-k 0 0))))

; Solver callbacks with auto-detection for 2D vs 3D slabs
(if (and (not run-te?) (not run-tm?) (not run-zeven?) (not run-zodd?))
    (if (and (defined? 'sz) (number? sz))
        (set! run-zeven? true)
        (set! run-te? true)))

(if (and (defined? 'sz) (not (number? sz)) run-zeven?)
    (begin (set! run-te? true) (set! run-zeven? false)))
(if (and (defined? 'sz) (not (number? sz)) run-zodd?)
    (begin (set! run-tm? true) (set! run-zodd? false)))

(if run-te? (run-solver-with-callbacks run-te))
(if run-tm? (run-solver-with-callbacks run-tm))
(if run-zeven? (run-solver-with-callbacks run-zeven))
(if run-zodd? (run-solver-with-callbacks run-zodd))
"""

def init_folder(target_dir: Union[str, Path] = ".") -> Tuple[Path, Path]:
    """
    Initializes a target directory with the modular Hydra configs/ hierarchy and main.ctl template.
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    dest_configs = target_path / "configs"
    ctl_file = target_path / "main.ctl"

    # Find source configs directory (in package src or repo root)
    src_configs = Path(__file__).parent / "configs"
    if not src_configs.is_dir():
        src_configs = Path(__file__).parents[2] / "configs"

    if not dest_configs.exists() and src_configs.is_dir():
        shutil.copytree(str(src_configs), str(dest_configs))
        print(f"Created modular Hydra configuration directory: '{dest_configs}'")
    elif dest_configs.exists():
        print(f"Configuration directory already exists: '{dest_configs}'")

    if not ctl_file.exists():
        ctl_file.write_text(MAIN_CTL_TEMPLATE)
        print(f"Created MPB control script: '{ctl_file}'")
    else:
        print(f"MPB control script already exists: '{ctl_file}'")

    return dest_configs, ctl_file

def main():
    parser = argparse.ArgumentParser(
        description="Initialize a directory with modular Hydra configs/ and main.ctl template."
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
    print("Initializing Photonic Crystal Simulation Workspace")
    print(f"Target Directory: '{target_dir}'")
    print("==================================================================")

    configs_dir, ctl_file = init_folder(target_dir)

    print("\nInitialization Complete!")
    print("Project Workspace Structure:")
    print(f"  {target_dir}/")
    print(f"  ├── main.ctl")
    print(f"  └── configs/")
    print(f"      ├── config.yaml")
    print(f"      ├── simulation/default.yaml")
    print(f"      ├── parameters/default.yaml")
    print(f"      ├── target/accidental_dirac.yaml")
    print(f"      ├── optimizer/bayesian.yaml")
    print(f"      └── postprocessing/default.yaml")
    print("\nYou can now run:")
    print(f"  cd {target_dir}")
    print(f"  uv run phc-bo")
    print(f"  uv run phc-bo parameters.fixed.h=0.35 simulation.parallel_workers=28")
    print(f"  uv run phc-bo -m parameters.fixed.h=0.30,0.35,0.40,0.45,0.50")
    print("==================================================================")

if __name__ == "__main__":
    main()
