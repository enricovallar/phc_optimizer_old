import subprocess
import os
import argparse
from pathlib import Path
from typing import Union, Optional


def rectify_h5_data(
    h5_path: Union[str, os.PathLike],
    output_path: Optional[Union[str, os.PathLike]] = None,
    rectangular: bool = True,
    resolution: int = 64,
    periods_x: int = 1,
    periods_y: int = 1,
    periods_z: int = 1,
    dataset_name: Optional[str] = None,
    version: str = "mpb/1.11.1",
) -> Path:
    """
    Rectify MPB HDF5 dataset (epsilon grid, e-field, h-field) using mpb-data tool.

    Transforms non-orthogonal/hexagonal basis grids into Cartesian rectangular grids,
    adjusts grid sampling resolution, and handles periodic supercell tiling.

    Parameters:
    -----------
    h5_path : str or PathLike
        Path to input HDF5 file (e.g. 'main-epsilon.h5' or 'e.k01.b01.h5').
    output_path : str or PathLike, optional
        Target path for rectified output HDF5 file. Defaults to <name>-rectified.h5.
    rectangular : bool, default True
        Transform non-orthogonal or hexagonal unit cells into rectangular Cartesian grid (-r).
    resolution : int, default 64
        Output grid resolution in grid points per unit cell period a (-n <resolution>).
    periods_x : int, default 1
        Number of unit cell periods to output along x (-x <periods_x>).
    periods_y : int, default 1
        Number of unit cell periods to output along y (-y <periods_y>).
    periods_z : int, default 1
        Number of unit cell periods to output along z (-z <periods_z>).
    dataset_name : str, optional
        Dataset name to process (-d <dataset_name>).
    version : str, default 'mpb/1.11.1'
        HPC MPB module version.

    Returns:
    --------
    Path : Path object pointing to the rectified HDF5 file.
    """
    input_file = Path(h5_path).resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input HDF5 file not found at '{input_file}'")

    if output_path is None:
        target_file = input_file.parent / f"{input_file.stem}-rectified{input_file.suffix}"
    else:
        target_file = Path(output_path).resolve()

    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Build mpb-data command line options
    flags = []
    if rectangular:
        flags.append("-r")
    if resolution:
        flags.extend(["-n", str(resolution)])
    if periods_x > 1:
        flags.extend(["-x", str(periods_x)])
    if periods_y > 1:
        flags.extend(["-y", str(periods_y)])
    if periods_z > 1:
        flags.extend(["-z", str(periods_z)])
    if dataset_name:
        flags.extend(["-d", str(dataset_name)])

    flags.extend(["-o", f'"{target_file}"', f'"{input_file}"'])
    flags_str = " ".join(flags)

    cmd = (
        f"source /dtu/sw/dcc/dcc-sw.bash && module load {version} && "
        f"mpb-data {flags_str}"
    )

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=target_file.parent,
        executable="/bin/bash"
    )

    if result.returncode != 0 or not target_file.is_file():
        raise RuntimeError(
            f"mpb-data execution failed with return code {result.returncode}.\n"
            f"Error output:\n{result.stderr}"
        )

    print(f"Rectified MPB HDF5 data saved to '{target_file}'")
    return target_file


def main() -> None:
    """CLI entry point for mpb-data rectifier tool."""
    parser = argparse.ArgumentParser(
        description="Rectify MPB HDF5 data grids (resolution, periodicity, rectangular Cartesian transformation)."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input MPB HDF5 file (e.g. main-epsilon.h5)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Target output HDF5 file path"
    )
    parser.add_argument(
        "-n", "--resolution",
        type=int,
        default=64,
        help="Output resolution (grid points per period a, default: 64)"
    )
    parser.add_argument(
        "-x", "--periods-x",
        type=int,
        default=1,
        help="Number of unit cell periods along x (default: 1)"
    )
    parser.add_argument(
        "-y", "--periods-y",
        type=int,
        default=1,
        help="Number of unit cell periods along y (default: 1)"
    )
    parser.add_argument(
        "-z", "--periods-z",
        type=int,
        default=1,
        help="Number of unit cell periods along z (default: 1)"
    )
    parser.add_argument(
        "--no-rect",
        action="store_true",
        help="Disable rectangular Cartesian grid transformation"
    )

    args = parser.parse_args()

    rectify_h5_data(
        h5_path=args.input,
        output_path=args.output,
        rectangular=not args.no_rect,
        resolution=args.resolution,
        periods_x=args.periods_x,
        periods_y=args.periods_y,
        periods_z=args.periods_z,
    )


if __name__ == "__main__":
    main()
