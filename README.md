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

# Specify custom target working directory and script:
uv run phc-runner --dir work --script main.ctl --cores 8
```

### CLI Arguments (`phc-runner --help`)

| Option | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--dir` | `-d` | Target working directory for execution & logs | Current Working Directory |
| `--script` | `-s` | Control file (`.ctl`) to run | `example.ctl` |
| `--cores` | `-c` | Number of MPI cores to use | `4` |
| `--no-mpi` | | Disable MPI parallel execution | False |

---

## Extracting Frequency Data (`.data` files)

The package automatically parses simulation logs (`output.out`) and extracts band frequencies and high-symmetry k-path labels into clean `.data` files (`tefreqs.data`, `tmfreqs.data`, `kpath_labels.data`).

You can also run the extractor manually on any MPB log file:

```bash
uv run phc-extractor -i work/output.out -o work/
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

# 1. Run MPB simulation
run_hpc(
    script="main.ctl",
    use_mpi=True,
    cores=4,
    wd="work"
)

# 2. Extract frequency data from output log manually if needed
data = extract_frequencies(output_path="work/output.out")
```
