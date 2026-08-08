import subprocess
import os
import argparse
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple


class MPBDataOptions:
    """Options for MPB data conversion matching phc_nzi.simulation_handler."""

    FLAG_MAP = {
        "rectify": "-r",
        "transpose": "-T",
        "pixellized": "-p",
    }

    VALUE_MAP = {
        "axis": "-e",
        "resolution": "-n",
        "phase": "-P",
        "dataset": "-d",
    }

    def __init__(
        self,
        rectify: bool = True,
        axis: Optional[int] = None,
        resolution: Optional[int] = None,
        periods: Optional[Union[int, Tuple[int, ...], List[int]]] = (3, 3, 1),
        phase: Optional[float] = None,
        transpose: bool = False,
        pixellized: bool = False,
        dataset: Optional[str] = None,
    ) -> None:
        """
        Initialize MPB data conversion options.

        Args:
            rectify: Whether to rectify the data (default: True)
            axis: Axis to extract (optional)
            resolution: Resolution to use (optional)
            periods: Number of periods in each direction (default: (3, 3, 1))
            phase: Phase to use for complex fields (optional)
            transpose: Whether to transpose the data (default: False)
            pixellized: Whether to use pixellized output (default: False)
            dataset: Dataset name to extract (optional)
        """
        self.rectify = rectify
        self.axis = axis
        self.resolution = resolution
        self.periods = periods
        self.phase = phase
        self.transpose = transpose
        self.pixellized = pixellized
        self.dataset = dataset

    def to_command_args(self) -> List[str]:
        """Convert options to command line arguments for mpb-data."""
        cmd = []

        for attr, flag in self.FLAG_MAP.items():
            if getattr(self, attr):
                cmd.append(flag)

        for attr, flag in self.VALUE_MAP.items():
            value = getattr(self, attr)
            if value is not None:
                cmd.extend([flag, str(value)])

        if self.periods is not None:
            if isinstance(self.periods, int):
                cmd.extend(["-m", str(self.periods)])
            elif isinstance(self.periods, (list, tuple)):
                if len(self.periods) >= 3:
                    cmd.extend(["-x", str(self.periods[0]),
                                "-y", str(self.periods[1]),
                                "-z", str(self.periods[2])])
                elif len(self.periods) == 2:
                    cmd.extend(["-x", str(self.periods[0]),
                                "-y", str(self.periods[1])])

        return cmd


class MPBDataConverter:
    """Handles conversion of MPB data files using the mpb-data utility."""

    def __init__(
        self,
        input_file: Union[str, os.PathLike],
        output_file: Union[str, os.PathLike],
        options: Optional[MPBDataOptions] = None,
        version: str = "mpb/1.11.1"
    ) -> None:
        self.input_file = str(Path(input_file).resolve())
        self.output_file = str(Path(output_file).resolve())
        self.options = options or MPBDataOptions()
        self.version = version

    def build_command(self) -> List[str]:
        """Build full bash command string for mpb-data execution."""
        cmd = [
            "source /dtu/sw/dcc/dcc-sw.bash &&",
            f"module load {self.version} &&",
            "mpb-data"
        ]
        cmd.extend(self.options.to_command_args())
        cmd.extend(["-o", f'"{self.output_file}"', f'"{self.input_file}"'])
        return cmd

    def run_conversion(self) -> str:
        """Execute mpb-data conversion command."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        cmd_list = self.build_command()
        full_cmd = " ".join(cmd_list)

        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash"
        )

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, full_cmd, result.stdout, result.stderr
            )

        return self.output_file


def rectify_h5_data(
    h5_path: Union[str, os.PathLike],
    output_path: Optional[Union[str, os.PathLike]] = None,
    options: Optional[MPBDataOptions] = None,
) -> Path:
    """
    Rectify MPB HDF5 dataset using MPBDataConverter and MPBDataOptions.

    Parameters:
    -----------
    h5_path : str or PathLike
        Path to input HDF5 file (e.g. 'main-epsilon.h5').
    output_path : str or PathLike, optional
        Target path for converted output HDF5 file. Defaults to <name>.converted.h5.
    options : MPBDataOptions, optional
        MPBDataOptions configuration. Defaults to MPBDataOptions(rectify=True, periods=(3, 3, 1)).

    Returns:
    --------
    Path : Path object pointing to converted HDF5 file.
    """
    input_file = Path(h5_path).resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input HDF5 file not found at '{input_file}'")

    if output_path is None:
        target_file = input_file.parent / f"{input_file.stem}.converted.h5"
    else:
        target_file = Path(output_path).resolve()

    opts = options or MPBDataOptions()
    converter = MPBDataConverter(input_file, target_file, opts)
    converter.run_conversion()

    print(f"Rectified MPB HDF5 data saved to '{target_file}'")
    return target_file


def main() -> None:
    """CLI entry point for mpb-data rectifier tool."""
    parser = argparse.ArgumentParser(
        description="Rectify MPB HDF5 data grids using mpb-data conversion options."
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
        default=None,
        help="Output resolution (grid points per period)"
    )
    parser.add_argument(
        "-p", "--periods",
        type=int,
        nargs="+",
        default=[3, 3, 1],
        help="Number of periods in each direction (default: 3 3 1)"
    )

    args = parser.parse_args()

    opts = MPBDataOptions(
        rectify=True,
        resolution=args.resolution,
        periods=tuple(args.periods)
    )

    rectify_h5_data(
        h5_path=args.input,
        output_path=args.output,
        options=opts
    )


if __name__ == "__main__":
    main()
