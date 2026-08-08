import subprocess
import os
import argparse
from pathlib import Path
from typing import Union, Dict, Any


def run_hpc(
    script: Union[str, os.PathLike],
    mpb_command_line_params: Dict[str, Any] = {},
    use_mpi: bool = True,
    cores: int = 4,
    version: str = "mpb/1.11.1",
    wd: Union[str, os.PathLike] = os.getcwd(),
    auto_extract: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run an MPB simulation using HPC module tools on DTU DCC cluster.

    Parameters:
    -----------
    script : str or PathLike
        Either a path to a .ctl script file or a string containing Scheme script content.
    mpb_command_line_params : dict
        Command line parameters to pass to MPB (e.g., {'num-bands': 10, 'resolution': 32}).
    use_mpi : bool
        Whether to use MPI parallel execution (mpirun -np <cores> mpb-mpi).
    cores : int
        Number of MPI cores to use.
    version : str
        Module version string for MPB on DCC (e.g. 'mpb/1.11.1').
    wd : str or PathLike
        Working directory for execution and output files.
    auto_extract : bool, default True
        Whether to automatically parse and extract frequency bands into .data files.
    """
    wd_path = Path(wd).resolve()
    wd_path.mkdir(parents=True, exist_ok=True)

    # Locate package root and ctl directory
    project_root = Path(__file__).resolve().parent.parent.parent
    package_ctl = project_root / "ctl"

    # Determine if script is a file path or script content
    script_str = str(script)
    if (wd_path / script_str).is_file():
        script_file = str(wd_path / script_str)
    elif (package_ctl / script_str).is_file():
        script_file = str(package_ctl / script_str)
    elif (Path.cwd() / script_str).is_file():
        script_file = str(Path.cwd() / script_str)
    elif (Path.cwd() / "ctl" / script_str).is_file():
        script_file = str(Path.cwd() / "ctl" / script_str)
    elif Path(script_str).is_file():
        script_file = script_str
    else:
        # If script is passed as raw Scheme string content, write to main.ctl
        script_file = str(wd_path / "main.ctl")
        with open(script_file, "w") as f:
            f.write(script_str)

    # Format MPB command line parameters (k=v pairs)
    params = " ".join(f"{k}={v}" for k, v in mpb_command_line_params.items())

    # Set MPI vs single-core command
    if use_mpi:
        mpb_cmd = f"mpirun -np {cores} mpb-mpi"
    else:
        mpb_cmd = "mpb"

    # Include init.ctl helper if present
    init_ctl = package_ctl / "init.ctl"
    init_arg = f"{init_ctl} " if init_ctl.is_file() else ""

    # Command string: ensure GUILE_LOAD_PATH includes package ctl directory, working directory, and cwd
    guile_paths = [
        str(package_ctl),
        str(wd_path),
        str(Path.cwd() / "ctl"),
        str(Path.cwd()),
    ]
    guile_load_path_str = ":".join(p for p in guile_paths if p) + ":$GUILE_LOAD_PATH"

    cmd = (
        f"export GUILE_LOAD_PATH=\"{guile_load_path_str}\" && "
        f"source /dtu/sw/dcc/dcc-sw.bash && module load {version} && "
        f"{mpb_cmd} {params} {init_arg}{script_file}"
    )

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=wd_path,
        executable="/bin/bash"
    )

    # Write error and output logs
    error_file = wd_path / "error.err"
    output_file = wd_path / "output.out"

    with open(error_file, "w") as f:
        f.write(result.stderr)
    with open(output_file, "w") as f:
        f.write(result.stdout)

    # Automatically extract frequency data if output log contains freqs
    if auto_extract and result.returncode == 0 and output_file.is_file():
        try:
            from .extractor import extract_frequencies
            extract_frequencies(output_path=output_file, output_dir=wd_path, save_data=True)
        except Exception as e:
            print(f"Note: Automatic data extraction notice: {e}")

    return result


def load_script(script_path: Union[str, os.PathLike]) -> str:
    """Load script text content from file."""
    path = Path(script_path)
    if not path.is_file():
        project_root = Path(__file__).resolve().parent.parent.parent
        if (project_root / "ctl" / script_path).is_file():
            path = project_root / "ctl" / script_path
        elif (Path.cwd() / "ctl" / script_path).is_file():
            path = Path.cwd() / "ctl" / script_path
    with open(path, "r") as f:
        return f.read()


def main() -> None:
    """CLI entry point for running MPB simulation script."""
    parser = argparse.ArgumentParser(
        description="Run MPB simulations using HPC module tools on DTU DCC cluster."
    )
    parser.add_argument(
        "-d", "--dir",
        type=str,
        default=os.getcwd(),
        help="Working directory for execution and output files (default: current working directory)"
    )
    parser.add_argument(
        "-s", "--script",
        type=str,
        default="example.ctl",
        help="MPB control file (.ctl) to run (default: example.ctl)"
    )
    parser.add_argument(
        "-c", "--cores",
        type=int,
        default=4,
        help="Number of MPI cores to use (default: 4)"
    )
    parser.add_argument(
        "--no-mpi",
        action="store_true",
        help="Disable MPI parallel execution"
    )

    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    script_path = args.script

    if input(f"Do you want to run the script '{script_path}' in directory '{target_dir}'? (y/n): ").strip().lower() == "y":
        print(f"Running MPB script '{script_path}' in directory '{target_dir}'...")
        res = run_hpc(
            script=script_path,
            mpb_command_line_params={},
            use_mpi=not args.no_mpi,
            cores=args.cores,
            version="mpb/1.11.1",
            wd=target_dir
        )
        if res.returncode == 0:
            print(f"Execution finished successfully! See '{target_dir / 'output.out'}' and '{target_dir / 'error.err'}' for results.")
        else:
            print(f"Execution failed with return code {res.returncode}. Check '{target_dir / 'error.err'}' for details.")


if __name__ == "__main__":
    main()
