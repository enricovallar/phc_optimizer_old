# phc-nzi

Modular MPB configuration and HPC runner package for photonic crystal simulations on DTU DCC.

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

# Specify custom target working directory and script:
uv run phc-runner --dir /path/to/directory --script example.ctl --cores 8
```

### CLI Arguments (`phc-runner --help`)

| Option | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--dir` | `-d` | Target working directory for execution & logs | Current Working Directory |
| `--script` | `-s` | Control file (`.ctl`) to run | `example.ctl` |
| `--cores` | `-c` | Number of MPI cores to use | `4` |
| `--no-mpi` | | Disable MPI parallel execution | False |

## Python API Usage

```python
from phc_nzi import run_hpc, load_script

# Run an MPB control file
run_hpc(
    script="example.ctl",
    mpb_command_line_params={"resolution": 32},
    use_mpi=True,
    cores=4,
    version="mpb/1.11.1",
    wd="/path/to/directory"
)
```
