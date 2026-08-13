"""
Bayesian Optimization for Photonic Crystal NZI & Dirac-like Cone Search
"""

import warnings
import os
import sys
import json
import time
import math
import yaml
import joblib
import argparse
import tempfile
import threading
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

from skopt import Optimizer
from skopt.space import Real, Integer

from phc_nzi.runner import run_hpc
from phc_nzi.extractor import extract_frequencies
from phc_nzi.symmetry import analyze_symmetries_from_log
from phc_nzi.plotter import plot_band_structure, plot_epsilon
from phc_nzi.geometry_check import check_slab_connectivity

warnings.filterwarnings("ignore", category=UserWarning, module="skopt")



def load_bo_config(config_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Loads and validates a Bayesian Optimization configuration file (YAML or JSON format).

    Parameters:
    -----------
    config_path : str or PathLike
        Path to the configuration file.

    Returns:
    --------
    dict
        Parsed configuration dictionary with defaults applied.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Optimization configuration file not found at '{config_path}'")

    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif path.suffix == ".json":
            config = json.load(f)
        else:
            # Attempt YAML parsing by default
            config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration file contents in '{config_path}'. Expected a dictionary.")

    # Apply standard defaults
    sim = config.get("simulation", {})
    sim.setdefault("ctl_script", "example.ctl")
    sim.setdefault("work_dir", "work")
    sim.setdefault("output_dir", "bo_output")
    sim.setdefault("cores", 4)
    sim.setdefault("only_gamma", True)
    sim.setdefault("local_workers", config.get("optimizer", {}).get("batch_size", 4))
    config["simulation"] = sim


    params = config.get("parameters", {})
    if not params:
        raise ValueError("No optimization parameters defined under 'parameters' in config file.")
    config["parameters"] = params

    config.setdefault("fixed_parameters", {})

    target = config.get("target", {})
    target.setdefault("symmetry_group", "C4v")
    target.setdefault("polarization", "te")
    target.setdefault("target_irreps", ["A_2", "E", "E"])
    target.setdefault("irrep_occurrences", [1, 1, 1])
    target.setdefault("min_band", 2)
    target.setdefault("degeneracy_tol", 0.005)
    target.setdefault("target_cost", 0.0025)
    target.setdefault("check_slab_connectivity", False)
    target.setdefault("enforce_connectivity", False)
    target.setdefault("epsilon_threshold", 1.1)
    target.setdefault("compute_group_velocity", False)
    target.setdefault("calculate_group_velocity", False)
    target.setdefault("delta_k", 0.01)
    config["target"] = target

    opt = config.get("optimizer", {})
    opt.setdefault("max_iterations", 20)
    n_pts = opt.get("initial_points", opt.get("n_initial_points", opt.get("initial_steps", 8)))
    opt["initial_points"] = n_pts
    opt.setdefault("initial_sampling", opt.get("initial_point_generator", "sobol"))
    opt.setdefault("model", opt.get("base_estimator", "GP"))
    opt.setdefault("acq_func", "LCB")
    opt.setdefault("acq_func_kwargs", {"kappa": 3.5})
    opt.setdefault("grid_evaluation", False)
    opt.setdefault("bypass_optimization", False)
    opt.setdefault("grid_resolution", None)
    opt.setdefault("objective_mode", "log")
    opt.setdefault("strategy", "cl_min")
    opt.setdefault("save_surrogate_freq", 0)  # Default 0: save only at the end
    opt.setdefault("random_state", 42)
    config["optimizer"] = opt




    return config


def _majority(iterable) -> bool:
    items = tuple(iterable)
    if not items:
        return False
    return sum(bool(x) for x in items) > len(items) / 2


def failsafe_irrep_mapping(
    target_irreps: List[str],
    degeneracy_tol: Optional[float],
    full_irrep_map: Dict[int, Tuple[str, float, float]],
    bands_to_check: List[int],
    log_file: Optional[Union[str, os.PathLike]] = None,
) -> List[Tuple[int, str, str]]:
    """
    FAILSAFE FOR DEGENERATE EIGENMODE MIXING:
    When exact band degeneracy occurs at Gamma, MPB returns linear combinations of eigenfunctions
    (state mixing), resulting in low confidence character projections or scrambled irrep labels.

    This function detects frequency-proximate band clusters within `degeneracy_tol` and relabels
    scrambled bands to match the target multiplet pattern if physical parity sequence rules hold.

    Returns a list of (band, old_irrep, new_irrep) tuples for each corrected band.
    """
    if degeneracy_tol is None or target_irreps is None:
        return []

    freqs = np.array([freq for _, _, freq in full_irrep_map.values()])
    n_targets = len(target_irreps)

    # PRE-CHECK: Skip correction if a valid consecutive set already matches target irreps
    if len(freqs) >= n_targets:
        for i in range(len(freqs) - n_targets + 1):
            is_degenerate = all(
                abs(freqs[i + k] - freqs[i + k + 1]) < degeneracy_tol for k in range(n_targets - 1)
            )
            if is_degenerate:
                current_irreps = [full_irrep_map[bands_to_check[i + k]][0] for k in range(n_targets)]
                if sorted(current_irreps) == sorted(target_irreps):
                    return []

    # Triplet/multiplet correction logic
    for i in range(len(freqs) - n_targets + 1):
        is_cluster = all(
            abs(freqs[i + k] - freqs[i + k + 1]) < degeneracy_tol for k in range(n_targets - 1)
        )
        if is_cluster:
            cluster_bands = bands_to_check[i : i + n_targets]
            cluster_irreps = [full_irrep_map[b][0] for b in cluster_bands]
            cluster_confs = [full_irrep_map[b][1] for b in cluster_bands]

            cond1 = sorted(cluster_irreps) != sorted(target_irreps)
            cond2 = _majority(conf < 0.85 for conf in cluster_confs)

            if cond1 and cond2:
                cond3 = any(irrep not in target_irreps for irrep in cluster_irreps)
                cond4 = any(conf >= 0.85 for conf in cluster_confs)
                if cond3 and cond4:
                    continue

                e_count_before = sum(
                    1
                    for j in range(i)
                    if full_irrep_map[bands_to_check[j]][0] is not None
                    and full_irrep_map[bands_to_check[j]][0].startswith("E")
                )

                if e_count_before % 2 == 0:
                    corrections = []
                    msg = (
                        f"[FAILSAFE RELABEL] Degenerate cluster at bands {cluster_bands} "
                        f"(freqs: {freqs[i:i+n_targets].tolist()}) relabeled from {cluster_irreps} "
                        f"to {target_irreps}\n"
                    )
                    if log_file:
                        with open(log_file, "a") as f:
                            f.write(msg)
                    else:
                        print(msg.strip())

                    for j, target_irrep in enumerate(target_irreps):
                        target_band = cluster_bands[j]
                        old_tuple = full_irrep_map[target_band]
                        old_irrep = old_tuple[0]
                        old_conf = old_tuple[1]
                        if old_irrep != target_irrep:
                            corrections.append((target_band, old_irrep, old_conf, target_irrep))
                        full_irrep_map[target_band] = (target_irrep, old_tuple[1], old_tuple[2])
                    return corrections

    return []


def find_bands_from_irreps(
    symmetries: List[Dict[str, Any]],
    parity: str,
    symmetry_group: str,
    target_irreps: List[str],
    irrep_occurrences: Optional[List[int]] = None,
    min_band: int = 2,
    degeneracy_tol: Optional[float] = None,
    log_file: Optional[Union[str, os.PathLike]] = None,
) -> Tuple[Optional[List[int]], Dict[int, Tuple[str, float, float]], Optional[str], List[Tuple[int, str, str]]]:
    """
    Dynamically maps requested target irreps to actual band indices based on extracted symmetry records.
    Excludes bands below min_band (default min_band=2 to avoid Band 1).

    Returns (bands, full_map, error_msg, corrections) where corrections is a list of
    (band, old_irrep, new_irrep) tuples from failsafe relabeling.
    """
    filtered = [
        r for r in symmetries
        if r.get("parity", "").lower() == parity.lower() and int(r.get("band", 0)) >= min_band
    ]

    if not filtered:
        return None, {}, f"No symmetry records found for parity '{parity}'.", []

    bands_to_check = [r["band"] for r in filtered]
    full_irrep_map = {
        r["band"]: (r["irrep"], float(r.get("confidence", 0.0)), float(r.get("freq", float("nan"))))
        for r in filtered
    }

    # Apply degeneracy failsafe
    corrections = failsafe_irrep_mapping(
        target_irreps, degeneracy_tol, full_irrep_map, bands_to_check, log_file=log_file
    )

    # Group bands by irrep label
    global_bands_by_irrep: Dict[str, List[int]] = {}
    for band, (irrep, conf, freq) in full_irrep_map.items():
        if irrep is None or irrep == "Unknown":
            continue
        global_bands_by_irrep.setdefault(irrep, []).append(band)

    # Slice sorted bands into representation dimensions (1D for A/B, 2D for E)
    modes_by_irrep: Dict[str, List[List[int]]] = {}
    for irrep, bands in global_bands_by_irrep.items():
        sorted_bands = sorted(bands)
        dim = 2 if irrep.startswith("E") else 1
        modes_by_irrep[irrep] = [sorted_bands[k : k + dim] for k in range(0, len(sorted_bands), dim)]

    occurrences = irrep_occurrences or [1] * len(target_irreps)
    dynamic_bands: List[int] = []
    used_bands = set()

    for irrep, occ in zip(target_irreps, occurrences):
        if irrep not in modes_by_irrep or len(modes_by_irrep[irrep]) < occ:
            error_msg = f"Missing occurrence {occ} for irrep '{irrep}'. Available modes: {modes_by_irrep}"
            return None, full_irrep_map, error_msg, corrections

        cluster = modes_by_irrep[irrep][occ - 1]
        assigned_band = None
        for b in cluster:
            if b not in used_bands:
                assigned_band = b
                break

        if assigned_band is None:
            error_msg = f"No unassigned band left in occurrence {occ} of '{irrep}'. Cluster: {cluster}"
            return None, full_irrep_map, error_msg, corrections

        dynamic_bands.append(assigned_band)
        used_bands.add(assigned_band)

    return dynamic_bands, full_irrep_map, None, corrections


class BayesianOptimizer:
    """
    Bayesian Optimization Controller for Photonic Crystal Band Structure Optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sim_cfg = config["simulation"]
        self.params_cfg = config["parameters"]
        self.fixed_params = config.get("fixed_parameters", {})
        self.target_cfg = config["target"]
        self.opt_cfg = config["optimizer"]

        self.param_names = list(self.params_cfg.keys())
        self.param_bounds = [self.params_cfg[k] for k in self.param_names]

        # Directory structure
        self.work_dir = Path(self.sim_cfg.get("work_dir", ".")).resolve()
        self.output_dir = self.work_dir / self.sim_cfg.get("output_dir", "bo_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_file = self.output_dir / "bo_trajectory.data"
        self.json_file = self.output_dir / "bo_data.json"
        self.irrep_log_file = self.output_dir / "bo_irreps.log"
        self.irreps_data_file = self.output_dir / "bo_irreps.data"
        self.model_file = self.output_dir / "bo_model.pkl"
        self.best_params_file = self.output_dir / "best_params.json"


        # Create skopt optimization space
        self.dimensions = []
        for name, bounds in zip(self.param_names, self.param_bounds):
            if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                self.dimensions.append(Integer(bounds[0], bounds[1], name=name))
            else:
                self.dimensions.append(Real(float(bounds[0]), float(bounds[1]), name=name))

        n_jobs = self.opt_cfg.get("n_jobs", self.sim_cfg.get("parallel_workers", 1))

        self.optimizer = Optimizer(
            dimensions=self.dimensions,
            base_estimator=self.opt_cfg.get("model", "GP"),
            n_initial_points=self.opt_cfg.get("initial_points", 8),
            initial_point_generator=self.opt_cfg.get("initial_sampling", "sobol"),
            acq_func=self.opt_cfg.get("acq_func", "LCB"),
            acq_func_kwargs=self.opt_cfg.get("acq_func_kwargs", {"kappa": 3.5}),
            n_jobs=n_jobs,
            random_state=self.opt_cfg.get("random_state", 42),
        )



        self.records: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.eval_counter = 0


    def _evaluate_single(
        self, args: Tuple[int, List[float]]
    ) -> Tuple[float, float, str, Dict[int, Any], List[int], List[Tuple[int, str, str]], str, Any, Dict[str, float]]:
        t_start = time.perf_counter()
        t_geom = 0.0
        t_run1 = 0.0
        t_run2 = 0.0

        gen, param_values = args
        param_dict = dict(zip(self.param_names, param_values))
        combined_params = {**self.fixed_params, **param_dict}
        combined_params["display_symmetry?"] = "true"

        with tempfile.TemporaryDirectory() as temp_dir:
            t_run1_0 = time.perf_counter()
            res = run_hpc(
                script=self._get_script_path(),
                mpb_command_line_params=combined_params,
                use_mpi=False,
                cores=self.sim_cfg.get("cores", 4),
                wd=temp_dir,
                auto_extract=True,
                auto_plot=False,
                only_gamma=self.sim_cfg.get("only_gamma", True),
                verbose=False,
            )
            t_run1 = time.perf_counter() - t_run1_0

            out_log = Path(temp_dir) / "output" / "output.out"
            output_text = ""
            if out_log.is_file():
                output_text = out_log.read_text()

            # Dielectric slab connectivity check (if enabled)
            conn_status = "NOT_CHECKED"
            enforce_conn = bool(
                self.target_cfg.get("check_slab_connectivity", False)
                or self.target_cfg.get("enforce_connectivity", False)
            )
            if enforce_conn:
                t_geom_0 = time.perf_counter()
                output_dir = Path(temp_dir) / "output"
                eps_h5 = output_dir / "main-epsilon.h5"
                if not eps_h5.exists():
                    h5_files = list(output_dir.glob("*-epsilon.h5"))
                    if h5_files:
                        eps_h5 = h5_files[0]

                if eps_h5.exists():
                    thresh = float(self.target_cfg.get("epsilon_threshold", 1.1))
                    min_neck = int(self.target_cfg.get("min_neck_width_px", 1))
                    is_conn, num_comp, conn_msg = check_slab_connectivity(
                        eps_h5, epsilon_threshold=thresh, check_pbc=True, min_neck_width_px=min_neck
                    )
                    t_geom = time.perf_counter() - t_geom_0
                    if not is_conn:
                        tracking_label = f"FAILED: Disconnected slab ({conn_msg})"
                        conn_status = f"FAILED ({conn_msg})"
                        t_total = time.perf_counter() - t_start
                        timing_dict = {"t_geom": t_geom, "t_run1": t_run1, "t_run2": 0.0, "t_total": t_total}
                        return 1.0, 0.0, tracking_label, {}, [], [], conn_status, None, timing_dict
                    else:
                        conn_status = f"PASSED ({conn_msg})"
                else:
                    t_geom = time.perf_counter() - t_geom_0
                    conn_status = "FAILED: Dielectric HDF5 grid file not found"
                    tracking_label = f"FAILED: {conn_status}"
                    t_total = time.perf_counter() - t_start
                    timing_dict = {"t_geom": t_geom, "t_run1": t_run1, "t_run2": 0.0, "t_total": t_total}
                    return 1.0, 0.0, tracking_label, {}, [], [], conn_status, None, timing_dict

            extracted_data = extract_frequencies(output_path=out_log, save_data=False, verbose=False)

            symmetry_records = analyze_symmetries_from_log(output_text, freq_data=extracted_data)

            target_irreps = self.target_cfg.get("target_irreps")
            pol = self.target_cfg.get("polarization", "te").lower()
            group = self.target_cfg.get("symmetry_group", "C4v")
            deg_tol = self.target_cfg.get("degeneracy_tol", 0.001)
            occurrences = self.target_cfg.get("irrep_occurrences", [1, 1, 1])
            min_b = self.target_cfg.get("min_band", 2)

            bypass_irrep = bool(
                self.target_cfg.get("bypass_irrep_identification", False)
                or self.target_cfg.get("bypass_symmetry", False)
                or self.target_cfg.get("use_static_bands", False)
                or (not target_irreps and (self.target_cfg.get("mode_indices") or self.target_cfg.get("bands")))
            )

            if not bypass_irrep and target_irreps:
                dynamic_bands, full_map, error_msg, corrections = find_bands_from_irreps(
                    symmetries=symmetry_records,
                    parity=pol,
                    symmetry_group=group,
                    target_irreps=target_irreps,
                    irrep_occurrences=occurrences,
                    min_band=min_b,
                    degeneracy_tol=deg_tol,
                    log_file=None,
                )

                if error_msg or not dynamic_bands:
                    tracking_label = f"FAILED: {error_msg}"
                    t_total = time.perf_counter() - t_start
                    timing_dict = {"t_geom": t_geom, "t_run1": t_run1, "t_run2": 0.0, "t_total": t_total}
                    return 1.0, 0.0, tracking_label, full_map, [], corrections, conn_status, None, timing_dict
                tracking_label = f"Mapped to bands {dynamic_bands}"
                target_bands = dynamic_bands
            else:
                static_modes = (
                    self.target_cfg.get("mode_indices")
                    or self.target_cfg.get("bands")
                    or self.target_cfg.get("target_bands")
                    or [2, 3, 4]
                )
                target_bands = [int(b) for b in static_modes]
                tracking_label = f"Mode indices {target_bands}"
                full_map = {
                    r["band"]: (r["irrep"], float(r.get("confidence", 0.0)), float(r.get("freq", float("nan"))))
                    for r in symmetry_records if r.get("parity", "").lower() == pol
                } if symmetry_records else {}
                corrections = []

            pol_key = f"{pol}freqs"
            if pol_key not in extracted_data or "headers" not in extracted_data[pol_key]:
                t_total = time.perf_counter() - t_start
                timing_dict = {"t_geom": t_geom, "t_run1": t_run1, "t_run2": 0.0, "t_total": t_total}
                return 1.0, 0.0, "FAILED: Frequency extraction empty", full_map, target_bands, corrections, conn_status, None, timing_dict

            rows = extracted_data[pol_key]["rows"]
            headers = extracted_data[pol_key]["headers"]

            # Compute cost at Gamma point
            gamma_row = rows[0]
            band_freqs = {}
            for col_idx, col_name in enumerate(headers):
                if "band_" in col_name:
                    try:
                        b_num = int(col_name.split("band_")[1])
                        band_freqs[b_num] = float(gamma_row[col_idx])
                    except (ValueError, IndexError):
                        pass

            idx_high = max(target_bands)
            idx_low = min(target_bands)
            freq_high = band_freqs.get(idx_high, 0.0)
            freq_low = band_freqs.get(idx_low, 0.0)

            freq_middle = (freq_high + freq_low) / 2.0
            if freq_middle <= 0:
                sorted_target = sorted(target_bands)
                idx_central = sorted_target[1] if len(sorted_target) >= 3 else sorted_target[0]
                freq_middle = band_freqs.get(idx_central, 1.0)

            raw_cost = abs(freq_high - freq_low)
            normalized_cost = raw_cost / freq_middle if freq_middle > 0 else raw_cost

            # Optional 2nd run: Group Velocity at small delta_k from Gamma for top target band
            vg_top_band = None
            compute_vg = bool(
                self.target_cfg.get("compute_group_velocity", False)
                or self.target_cfg.get("calculate_group_velocity", False)
            )

            if compute_vg and target_bands:
                top_band = max(target_bands)
                delta_k = float(self.target_cfg.get("delta_k", 0.01))

                vg_params = dict(combined_params)
                vg_params["only_gamma?"] = "false"
                vg_params["display_group_velocity?"] = "true"
                vg_params["delta_k"] = delta_k
                vg_params["delta_k_mode?"] = "true"

                with tempfile.TemporaryDirectory() as vg_temp_dir:
                    t_run2_0 = time.perf_counter()
                    res_vg = run_hpc(
                        script=self._get_script_path(),
                        mpb_command_line_params=vg_params,
                        use_mpi=False,
                        cores=self.sim_cfg.get("cores", 4),
                        wd=vg_temp_dir,
                        auto_extract=True,
                        auto_plot=False,
                        only_gamma=False,
                        verbose=False,
                    )
                    t_run2 = time.perf_counter() - t_run2_0
                    vg_log = Path(vg_temp_dir) / "output" / "output.out"
                    if vg_log.is_file():
                        from phc_nzi.extractor import extract_group_velocities
                        vg_data = extract_group_velocities(output_path=vg_log, save_data=False, verbose=False)
                        flat_recs = vg_data.get("flat_records", [])
                        for rec_v in flat_recs:
                            if rec_v.get("parity", "").lower() == pol and int(rec_v.get("band", 0)) == top_band:
                                vg_top_band = float(rec_v.get("vg_mag", 0.0))
                                break

            t_total = time.perf_counter() - t_start
            timing_dict = {"t_geom": t_geom, "t_run1": t_run1, "t_run2": t_run2, "t_total": t_total}
            return normalized_cost, freq_middle, tracking_label, full_map, target_bands, corrections, conn_status, vg_top_band, timing_dict

    def objective(self, gen: int, param_values: List[float]) -> float:
        cost, freq_dirac, label, full_map, target_bands, corrections, conn_status, vg_top_band, timing_dict = self._evaluate_single((gen, param_values))

        with self.lock:
            self.eval_counter += 1
            eval_idx = self.eval_counter

            debug_t = bool(self.sim_cfg.get("debug_timing", False) or self.opt_cfg.get("debug_timing", False))
            param_dict = dict(zip(self.param_names, param_values))
            corr_str = ", ".join(f"Band {b}: {old} (conf = {old_conf:.3f}) -> {new}" for b, old, old_conf, new in corrections) if corrections else ""
            rec = {
                "eval_number": eval_idx,
                "generation": gen,
                "params": param_dict,
                "raw_cost": cost,
                "freq_dirac": freq_dirac,
                "group_velocity": vg_top_band,
                "status": label,
                "connectivity": conn_status,
                "target_bands": target_bands,
                "timing": timing_dict,
                "corrected": bool(corrections),
                "corrections": [{"band": b, "old": old, "old_conf": old_conf, "new": new} for b, old, old_conf, new in corrections],
                "full_map": {
                    str(b): {
                        "irrep": irrep_info[0],
                        "confidence": float(irrep_info[1]) if not np.isnan(irrep_info[1]) else 0.0,
                        "freq": float(irrep_info[2]) if not np.isnan(irrep_info[2]) else 0.0,
                        "selected": (b in target_bands),
                        "target_num": (target_bands.index(b) + 1 if b in target_bands else None),
                    }
                    for b, irrep_info in sorted(full_map.items())
                } if full_map else {},
            }
            self.records.append(rec)

            # 1. Write tabular trajectory log
            with open(self.data_file, "a") as f:
                p_str = ", ".join(f"{k}: {v:.6f}" for k, v in param_dict.items())
                t_str = f" | time: {timing_dict['t_total']:.2f}s (geom: {timing_dict['t_geom']:.3f}s, run1: {timing_dict['t_run1']:.2f}s, run2: {timing_dict['t_run2']:.2f}s)" if debug_t else ""
                f.write(
                    f"Eval: {eval_idx:03d} | Gen: {gen:02d} | {p_str} | cost: {cost:.6f} | freq_dirac: {freq_dirac:.6f}{t_str} | [{label}]\n"
                )

            # 2. Write detailed human-readable irreps log (bo_irreps.log)
            with open(self.irrep_log_file, "a") as f:
                f.write("=" * 90 + "\n")
                f.write(f"EVALUATION #{eval_idx:03d} (Generation {gen:02d})\n")
                p_log_str = ", ".join(f"{k} = {v:.6f}" for k, v in param_dict.items())
                f.write(f"Parameters   : {p_log_str}\n")
                f.write(f"Cost         : {cost:.6f} | Dirac Freq: {freq_dirac:.6f} | {label}\n")
                if conn_status != "NOT_CHECKED":
                    f.write(f"Connectivity : {conn_status}\n")
                if debug_t:
                    f.write(f"Timing       : total {timing_dict['t_total']:.2f}s (geom {timing_dict['t_geom']:.3f}s, run1 {timing_dict['t_run1']:.2f}s, run2 {timing_dict['t_run2']:.2f}s)\n")
                if vg_top_band is not None:
                    top_b_idx = max(target_bands) if target_bands else 0
                    d_k = self.target_cfg.get("delta_k", 0.01)
                    f.write(f"Top Band vg  : {vg_top_band:.6f} c (Band #{top_b_idx} at delta_k = {d_k:.4f})\n")
                if corrections:
                    f.write(f"CORRECTED    : {corr_str}\n")
                f.write("-" * 90 + "\n")
                if full_map:
                    corr_lookup = {b: (old, old_conf) for b, old, old_conf, new in corrections}
                    for b in sorted(full_map.keys()):
                        irrep_name, conf, freq = full_map[b]
                        relab_str = f" [relabeled from {corr_lookup[b][0]} (old conf = {corr_lookup[b][1]:.3f})]" if b in corr_lookup else ""
                        if b in target_bands:
                            t_num = target_bands.index(b) + 1
                            f.write(f"* Band {b:2d} [Target #{t_num}]: {irrep_name:<8s} (freq = {freq:.6f}, conf = {conf:.3f}){relab_str}\n")
                        else:
                            f.write(f"  Band {b:2d}           : {irrep_name:<8s} (freq = {freq:.6f}, conf = {conf:.3f}){relab_str}\n")
                else:
                    f.write("  No irrep map available.\n")
                f.write("=" * 90 + "\n\n")

            # 3. Write structured tabular data file (bo_irreps.data)
            with open(self.irreps_data_file, "a") as f:
                if eval_idx == 1:
                    p_hdr = " ".join(f"{k:<10s}" for k in self.param_names)
                    f.write(f"# Eval  Gen  {p_hdr} Cost       Dirac_Freq  Target_Bands           Corrected  All_Bands_Irreps\n")
                p_vals_str = " ".join(f"{v:<10.6f}" for v in param_values)
                t_bands_str = ",".join(f"{b}*[#{target_bands.index(b)+1}]" for b in target_bands) if target_bands else "None"
                corrected_flag = "YES" if corrections else "NO"
                def _band_label(b):
                    if b in target_bands:
                        return f"{b}*:{full_map[b][0]}"
                    else:
                        return f"{b}:{full_map[b][0]}"
                all_irreps_str = ", ".join(_band_label(b) for b in sorted(full_map.keys())) if full_map else "None"
                f.write(f"{eval_idx:<6d} {gen:<4d} {p_vals_str} {cost:<10.6f} {freq_dirac:<11.6f} {t_bands_str:<22s} {corrected_flag:<10s} {all_irreps_str}\n")



        # 4. Save JSON checkpoint
        with open(self.json_file, "w") as f:
            json.dump(self.records, f, indent=2)

        target_cost = self.target_cfg.get("target_cost")
        effective_cost = (
            target_cost if (target_cost is not None and cost < target_cost) else cost
        )

        mode = self.opt_cfg.get("objective_mode", "log")
        if mode == "log":
            return float(np.log10(max(effective_cost, 1e-12)))
        return float(effective_cost)


    def _generate_grid_points(self) -> List[List[float]]:
        """Generates a uniform grid of parameter points across param_bounds."""
        grid_res = self.opt_cfg.get("grid_resolution")
        n_dims = len(self.param_names)

        if isinstance(grid_res, int):
            res_per_dim = [grid_res] * n_dims
        elif isinstance(grid_res, (list, tuple)) and len(grid_res) == n_dims:
            res_per_dim = [int(r) for r in grid_res]
        else:
            total_pts = self.opt_cfg.get("initial_points", 64)
            n_per_dim = max(2, int(round(total_pts ** (1.0 / n_dims))))
            res_per_dim = [n_per_dim] * n_dims

        coords = [
            np.linspace(float(b[0]), float(b[1]), n_pts)
            for b, n_pts in zip(self.param_bounds, res_per_dim)
        ]
        if n_dims == 1:
            return [[float(x)] for x in coords[0]]
        mesh = np.meshgrid(*coords, indexing="ij")
        points = np.vstack([m.ravel() for m in mesh]).T
        return [[float(val) for val in pt] for pt in points]

    def run(self) -> Dict[str, Any]:
        """
        Executes the main Bayesian Optimization loop or Uniform Grid Evaluation.
        """
        n_initial_pts = self.opt_cfg.get("initial_points", 8)
        batch_size = max(1, self.opt_cfg.get("batch_size", 4))
        bo_generations = self.opt_cfg.get("max_iterations", 10)
        strategy = self.opt_cfg.get("strategy", "cl_min")
        workers = self.sim_cfg.get("local_workers", batch_size)
        surrogate_freq = int(self.opt_cfg.get("save_surrogate_freq", 0))

        # Clear previous trajectory files
        self.data_file.write_text("")
        self.irrep_log_file.write_text("")
        self.irreps_data_file.write_text("")

        is_grid_mode = bool(
            self.opt_cfg.get("grid_evaluation", False)
            or self.opt_cfg.get("bypass_optimization", False)
            or self.opt_cfg.get("grid_search", False)
        )

        if is_grid_mode:
            grid_pts = self._generate_grid_points()
            total_evals = len(grid_pts)
            n_batches = math.ceil(total_evals / batch_size)

            print("==================================================================")
            print("Starting Photonic Crystal Uniform Grid Evaluation (Bypass BO)")
            print(f"Working Directory:   '{self.work_dir}'")
            print(f"Output Directory:    '{self.output_dir}'")
            print(f"Parameters:          {self.param_names}")
            print(f"Target Irreps:       {self.target_cfg.get('target_irreps')} ({self.target_cfg.get('symmetry_group')})")
            print(f"Grid Sampling:       {total_evals} points ({n_batches} batches)")
            print(f"Surrogate Model:     {self.opt_cfg.get('model', 'GP')}")
            print(f"Parallel Workers:    {workers} concurrent evaluation workers")
            print(f"Only Gamma:          {self.sim_cfg.get('only_gamma', True)}")
            print("==================================================================")

            pbar = tqdm(total=n_batches, desc="Grid Progress", unit="batch")
            for b_idx in range(n_batches):
                batch_start = b_idx * batch_size
                x_batch = grid_pts[batch_start : batch_start + batch_size]
                gen = b_idx + 1

                if workers > 1:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = [executor.submit(self.objective, gen, x) for x in x_batch]
                        y_batch = [f.result() for f in futures]
                else:
                    y_batch = [self.objective(gen, x) for x in x_batch]

                self.optimizer.tell(x_batch, y_batch)
                best_score = min(self.optimizer.yi)

                pbar.set_postfix({"batch": f"{b_idx+1}/{n_batches}", "best_cost": f"{best_score:.6f}"})
                pbar.update(1)

                if surrogate_freq > 0 and (b_idx + 1) % surrogate_freq == 0:
                    self._plot_surrogate_map(verbose=False)

            pbar.close()

        else:
            initial_generations = math.ceil(n_initial_pts / batch_size)
            total_generations = initial_generations + bo_generations

            print("==================================================================")
            print("Starting Photonic Crystal Bayesian Optimization")
            print(f"Working Directory:   '{self.work_dir}'")
            print(f"Output Directory:    '{self.output_dir}'")
            print(f"Parameters:          {self.param_names}")
            print(f"Target Irreps:       {self.target_cfg.get('target_irreps')} ({self.target_cfg.get('symmetry_group')})")
            print(f"Initial Sampling:    {self.opt_cfg.get('initial_sampling', 'sobol')} ({n_initial_pts} points = {initial_generations} initial gens)")
            print(f"Guided BO Phase:     {self.opt_cfg.get('model', 'GP')} ({bo_generations} GP gens, Batch Size: {batch_size})")
            print(f"Total Generations:   {total_generations} ({total_generations * batch_size} max evals)")
            print(f"Parallel Workers:    {workers} concurrent evaluation workers")
            print(f"Only Gamma:          {self.sim_cfg.get('only_gamma', True)}")
            print("==================================================================")

            pbar = tqdm(total=total_generations, desc="BO Progress", unit="gen")
            for gen in range(total_generations):
                t_gen_start = time.perf_counter()

                t_ask_0 = time.perf_counter()
                x_batch = self.optimizer.ask(n_points=batch_size, strategy=strategy)
                t_ask = time.perf_counter() - t_ask_0

                t_workers_0 = time.perf_counter()
                if workers > 1:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = [
                            executor.submit(self.objective, gen + 1, x) for x in x_batch
                        ]
                        y_batch = [f.result() for f in futures]
                else:
                    y_batch = [self.objective(gen + 1, x) for x in x_batch]
                t_workers = time.perf_counter() - t_workers_0

                t_tell_0 = time.perf_counter()
                self.optimizer.tell(x_batch, y_batch)
                t_tell = time.perf_counter() - t_tell_0

                t_gen_total = time.perf_counter() - t_gen_start
                best_score = min(self.optimizer.yi)
                current_evals = len(self.optimizer.Xi)

                debug_t = bool(self.sim_cfg.get("debug_timing", False) or self.opt_cfg.get("debug_timing", False))
                if debug_t:
                    gen_recs = self.records[-len(x_batch):]
                    geom_times = [r["timing"]["t_geom"] for r in gen_recs if "timing" in r]
                    run1_times = [r["timing"]["t_run1"] for r in gen_recs if "timing" in r]
                    run2_times = [r["timing"]["t_run2"] for r in gen_recs if "timing" in r]

                    avg_geom = float(np.mean(geom_times)) if geom_times else 0.0
                    avg_r1 = float(np.mean(run1_times)) if run1_times else 0.0
                    min_r1 = float(np.min(run1_times)) if run1_times else 0.0
                    max_r1 = float(np.max(run1_times)) if run1_times else 0.0

                    avg_r2 = float(np.mean(run2_times)) if run2_times else 0.0
                    min_r2 = float(np.min(run2_times)) if run2_times else 0.0
                    max_r2 = float(np.max(run2_times)) if run2_times else 0.0

                    summary_lines = [
                        "=" * 70,
                        f"GENERATION {gen+1:02d} TIMING SUMMARY ({len(x_batch)} Parallel Workers)",
                        "-" * 70,
                        f"Geometry Continuity Check : avg {avg_geom:.3f} s",
                        f"MPB Run 1 (Gamma & Irreps): avg {avg_r1:.2f} s | min {min_r1:.2f} s | max {max_r1:.2f} s",
                        f"MPB Run 2 (Group Velocity): avg {avg_r2:.2f} s | min {min_r2:.2f} s | max {max_r2:.2f} s",
                        f"ML Acquisition Search (ask):     {t_ask:.3f} s",
                        f"ML Model Update (tell)     :     {t_tell:.3f} s",
                        f"Generation Total Wall Time :     {t_gen_total:.2f} s",
                        "=" * 70 + "\n",
                    ]
                    summary_text = "\n".join(summary_lines)
                    tqdm.write(summary_text)
                    with open(self.irrep_log_file, "a") as f:
                        f.write(summary_text)

                if gen < initial_generations:
                    phase_str = f"Initial Sobol ({current_evals}/{n_initial_pts} pts)"
                else:
                    bo_idx = gen - initial_generations + 1
                    phase_str = f"Guided GP ({bo_idx}/{bo_generations})"

                pbar.set_postfix({"phase": phase_str, "best_cost": f"{best_score:.6f}"})
                pbar.update(1)

                if surrogate_freq > 0 and (gen + 1) % surrogate_freq == 0:
                    self._plot_surrogate_map(verbose=False)

            pbar.close()



        # Save optimizer model checkpoint
        joblib.dump(self.optimizer, self.model_file)

        best_idx = int(np.argmin(self.optimizer.yi))
        best_params = dict(zip(self.param_names, [float(x) for x in self.optimizer.Xi[best_idx]]))
        best_record = self.records[best_idx] if best_idx < len(self.records) else {}

        print("==================================================================")
        print("Optimization Complete!")
        print(f"Optimal Parameters: {best_params}")
        print(f"Optimal Cost:       {best_record.get('raw_cost', float('nan')):.6f}")
        print("==================================================================")

        # Write best parameters file
        best_summary = {
            "optimal_parameters": best_params,
            "fixed_parameters": self.fixed_params,
            "raw_cost": best_record.get("raw_cost"),
            "freq_dirac": best_record.get("freq_dirac"),
            "surrogate_score": float(self.optimizer.yi[best_idx]),
        }
        with open(self.best_params_file, "w") as f:
            json.dump(best_summary, f, indent=2)

        # Plot convergence curve, surrogate map & group velocity map
        self._plot_convergence()
        self._plot_surrogate_map()
        self._plot_group_velocity_map()

        # Run final full k-path validation simulation
        self._run_validation(best_params)

        return best_summary


    def _plot_convergence(self) -> None:
        if not self.records:
            return
        costs = [r["raw_cost"] for r in self.records]
        best_so_far = np.minimum.accumulate(costs)
        n_evals = len(costs)

        n_initial = self.opt_cfg.get("initial_points", 8)
        n_initial_evals = min(n_initial, n_evals)


        plt.figure(figsize=(9, 5.5))
        eval_indices = np.arange(1, n_evals + 1)

        # Distinguish initial sampling phase vs active BO phase
        if n_initial_evals > 0:
            plt.scatter(
                eval_indices[:n_initial_evals],
                costs[:n_initial_evals],
                color="royalblue",
                s=40,
                zorder=4,
                label=f"Initial Sampling ({self.opt_cfg.get('initial_sampling', 'sobol')})",
            )
            plt.axvspan(0.5, n_initial_evals + 0.5, color="royalblue", alpha=0.08)
            plt.axvline(x=n_initial_evals + 0.5, color="gray", linestyle="--", alpha=0.7, label="BO Phase Start")

        if n_evals > n_initial_evals:
            plt.scatter(
                eval_indices[n_initial_evals:],
                costs[n_initial_evals:],
                color="crimson",
                marker="s",
                s=40,
                zorder=4,
                label="Active BO Evaluations (GP)",
            )
            plt.axvspan(n_initial_evals + 0.5, n_evals + 0.5, color="crimson", alpha=0.05)

        plt.plot(eval_indices, costs, "-", color="gray", alpha=0.3)
        plt.plot(eval_indices, best_so_far, "r-", linewidth=2.5, zorder=5, label="Best Cost So Far")

        plt.yscale("log")
        plt.xlabel("Evaluation #", fontsize=11)
        plt.ylabel("Dirac Cone Gap Cost", fontsize=11)
        plt.title("Bayesian Optimization Convergence (Initial vs. Guided BO)", fontsize=12, fontweight="bold")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend(loc="upper right")
        plt.tight_layout()

        conv_file = self.output_dir / "bo_convergence.png"
        plt.savefig(conv_file, dpi=200)
        plt.close()
        print(f"Saved convergence plot to '{conv_file}'")

    def _plot_surrogate_map(self, verbose: bool = True) -> None:
        """
        Generates and saves a 2-panel figure matching pillars_c6v_1.ipynb styling:
        - Plot 1 (Left): Raw optimization evaluated points colored by FOM (1/Cost) using LogNorm and 'cool' colormap.
        - Plot 2 (Right): GP Surrogate model predicted FOM (1/Cost) contour landscape with LogNorm and 'cool' colormap,
          overlaid with sampled points and optimal point star.
        """
        import matplotlib.colors as mcolors
        from matplotlib.ticker import FormatStrFormatter, LogFormatterSciNotation, LogLocator

        if not hasattr(self.optimizer, "models") or not self.optimizer.models:
            return

        model = self.optimizer.models[-1]
        surrogate_file = self.output_dir / "bo_surrogate_map.png"

        if len(self.param_names) == 2:
            p1_name, p2_name = self.param_names[0], self.param_names[1]
            b1 = self.param_bounds[0]
            b2 = self.param_bounds[1]

            x1 = np.linspace(float(b1[0]), float(b1[1]), 200)
            x2 = np.linspace(float(b2[0]), float(b2[1]), 200)
            X1, X2 = np.meshgrid(x1, x2)
            grid_points = np.c_[X1.ravel(), X2.ravel()]

            # Transform grid points to normalized search space expected by the GP estimator
            try:
                grid_points_transformed = self.optimizer.space.transform(grid_points.tolist())
            except Exception:
                grid_points_transformed = grid_points

            # Evaluate surrogate model predictions with std (uncertainty) to compute Expectation Value E[C]
            mode = self.opt_cfg.get("objective_mode", "log")
            try:
                if hasattr(model, "predict"):
                    try:
                        mu_grid, std_grid = model.predict(grid_points_transformed, return_std=True)
                    except Exception:
                        mu_grid = model.predict(grid_points_transformed)
                        std_grid = np.zeros_like(mu_grid)
                else:
                    mu_grid = np.zeros(len(grid_points))
                    std_grid = np.zeros_like(mu_grid)

                mu_grid = mu_grid.reshape(X1.shape)
                std_grid = std_grid.reshape(X1.shape)
            except Exception as e:
                if verbose:
                    print(f"Note: Could not compute surrogate predictions: {e}")
                return

            Xi = np.array(self.optimizer.Xi)
            yi = np.array(self.optimizer.yi)

            # Extract raw evaluated costs by matching Xi parameter values to records unambiguously
            raw_costs_list = []
            for idx, x_pt in enumerate(Xi):
                matched_cost = None
                for r in self.records:
                    r_pt = [r["params"][name] for name in self.param_names if name in r["params"]]
                    if len(r_pt) == len(x_pt) and np.allclose(r_pt, x_pt, atol=1e-5):
                        matched_cost = r["raw_cost"]
                        break
                if matched_cost is not None:
                    raw_costs_list.append(matched_cost)
                elif mode == "log":
                    raw_costs_list.append(10 ** float(yi[idx]))
                else:
                    raw_costs_list.append(float(yi[idx]))

            raw_costs = np.array(raw_costs_list)

            # Compute Expectation Value of Cost E[C] and its inverse E[C]^-1
            # For y = log10(C), C is Log-Normal: E[C] = 10^(mu + (ln(10)/2) * std^2)
            if mode == "log":
                exp_c_grid = 10 ** (mu_grid + (np.log(10) / 2.0) * (std_grid ** 2))
            else:
                exp_c_grid = mu_grid

            fom_raw = 1.0 / np.maximum(raw_costs, 1e-12)
            predicted_e_c_inv = 1.0 / np.maximum(exp_c_grid, 1e-12)

            # Compute shared LogNorm scale bounds using robust percentiles
            all_fom_vals = np.r_[fom_raw, predicted_e_c_inv.ravel()]
            finite_pos = all_fom_vals[np.isfinite(all_fom_vals) & (all_fom_vals > 0)]

            if finite_pos.size > 0:
                vmin = max(float(np.percentile(finite_pos, 1)), 1e-12)
                vmax = float(np.percentile(finite_pos, 99))
                if vmax <= vmin:
                    vmax = vmin * 10
            else:
                vmin, vmax = 1.0, 1000.0

            log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

            target_irreps = self.target_cfg.get("target_irreps", ["A_2", "E", "E"])
            target_str = " - ".join(dict.fromkeys(target_irreps))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0))

            # ---------------------------------------------------------
            # Plot 1 (Left): Raw Optimization Data
            # ---------------------------------------------------------
            sc_raw = ax1.scatter(
                Xi[:, 0], Xi[:, 1], c=fom_raw, cmap="cool", norm=log_norm, s=35, edgecolors="black", linewidths=0.5, marker="o", zorder=4
            )

            best_idx = int(np.argmin(yi))
            best_x = Xi[best_idx]
            ax1.scatter(best_x[0], best_x[1], c="gold", edgecolors="black", marker="*", s=220, zorder=6, label="Optimal Point")

            ax1.set_xlabel(f"${p1_name}/a$", fontsize=11)
            ax1.set_ylabel(f"${p2_name}/a$", fontsize=11)
            ax1.set_title(f"(a) Evaluated Points ({target_str})", fontsize=12, fontweight="bold")
            ax1.set_xlim(float(b1[0]), float(b1[1]))
            ax1.set_ylim(float(b2[0]), float(b2[1]))
            ax1.set_aspect("equal", adjustable="box")
            ax1.grid(alpha=0.4, linestyle="--")

            # ---------------------------------------------------------
            # Plot 2 (Right): GP Surrogate Expectation Map E[C]^-1
            # ---------------------------------------------------------
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 100)
            heatmap = ax2.contourf(
                X1, X2, predicted_e_c_inv, levels=levels, cmap="cool", norm=log_norm, extend="both"
            )

            ax2.scatter(Xi[:, 0], Xi[:, 1], c="white", edgecolors="black", s=25, alpha=0.7, label="Explored points", zorder=5)
            ax2.scatter(best_x[0], best_x[1], c="gold", edgecolors="black", marker="*", s=220, zorder=6, label="Optimal Point")

            ax2.set_xlabel(f"${p1_name}/a$", fontsize=11)
            ax2.set_ylabel(f"${p2_name}/a$", fontsize=11)
            ax2.set_title(f"(b) {target_str} Tracking", fontsize=12, fontweight="bold")
            ax2.set_xlim(float(b1[0]), float(b1[1]))
            ax2.set_ylim(float(b2[0]), float(b2[1]))
            ax2.set_aspect("equal", adjustable="box")
            ax2.grid(alpha=0.4, linestyle="--")

            # Shared tick format
            for ax in (ax1, ax2):
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # Shared Colorbar on the right matching pillars_c6v_1.ipynb
            cbar = fig.colorbar(heatmap, ax=[ax1, ax2], fraction=0.035, pad=0.04, extend="both", format=LogFormatterSciNotation())
            cbar.set_label(r"1/$\mathrm{C}$", fontsize=13)
            cbar.ax.tick_params(labelsize=9)
            cbar.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
            cbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())
            cbar.ax.tick_params(which="minor", length=4)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels1 + labels2, handles1 + handles2))
            fig.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, facecolor="white", edgecolor="black")

            plt.savefig(surrogate_file, dpi=200, bbox_inches="tight")
            plt.close()
            if verbose:
                print(f"Saved surrogate map plot to '{surrogate_file}'")




        else:
            try:
                from skopt.plots import plot_objective
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_objective(self.optimizer, ax=ax)
                plt.tight_layout()
                plt.savefig(surrogate_file, dpi=200)
                plt.close()
                if verbose:
                    print(f"Saved surrogate map plot to '{surrogate_file}'")
            except Exception as e:
                if verbose:
                    print(f"Note: Could not generate multi-dimensional surrogate plot: {e}")



    def _plot_group_velocity_map(self, verbose: bool = True) -> None:
        """
        Generates and saves a 2-panel figure of the top target band group velocity vg (in units of c)
        computed at a small delta_k from Gamma.
        """
        import matplotlib.colors as mcolors
        from matplotlib.ticker import FormatStrFormatter

        vg_records = [r for r in self.records if r.get("group_velocity") is not None]
        if not vg_records:
            return

        fig_file = self.output_dir / "bo_group_velocity_map.png"
        data_file = self.output_dir / "bo_group_velocity.data"

        # Write tabular group velocity data file
        with open(data_file, "w") as f:
            p_hdr = " ".join(f"{k:<10s}" for k in self.param_names)
            f.write(f"# Eval  Gen  {p_hdr} Top_Band  vg_mag (c)   Cost\n")
            for r in vg_records:
                p_str = " ".join(f"{float(r['params'][k]):<10.6f}" for k in self.param_names)
                top_b = max(r.get("target_bands", [6])) if r.get("target_bands") else 0
                vg_v = float(r.get("group_velocity", 0.0))
                c_v = float(r.get("raw_cost", 0.0))
                f.write(f"{r['eval_number']:<6d} {r['generation']:<4d} {p_str} {top_b:<9d} {vg_v:<12.6f} {c_v:<10.6f}\n")

        if len(self.param_names) == 2:
            p1_name, p2_name = self.param_names[0], self.param_names[1]
            b1 = self.param_bounds[0]
            b2 = self.param_bounds[1]

            Xi_vg = np.array([[r["params"][p1_name], r["params"][p2_name]] for r in vg_records])
            yi_vg = np.array([r["group_velocity"] for r in vg_records])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0))

            top_b_num = max(vg_records[0].get("target_bands", [6])) if vg_records[0].get("target_bands") else "Top"
            delta_k = self.target_cfg.get("delta_k", 0.01)

            # Fit surrogate on group velocity using configured model type (GP, RF, ET, GBRT)
            try:
                model_type = str(self.opt_cfg.get("model", "GP")).upper()
                if model_type in ("RF", "RANDOM_FOREST"):
                    from skopt.learning import RandomForestRegressor
                    reg_vg = RandomForestRegressor(random_state=42)
                elif model_type in ("ET", "EXTRA_TREES"):
                    from skopt.learning import ExtraTreesRegressor
                    reg_vg = ExtraTreesRegressor(random_state=42)
                elif model_type in ("GBRT", "GRADIENT_BOOSTING"):
                    from skopt.learning import GradientBoostingQuantileRegressor
                    reg_vg = GradientBoostingQuantileRegressor(random_state=42)
                else:
                    from skopt.learning import GaussianProcessRegressor
                    reg_vg = GaussianProcessRegressor(random_state=42)

                reg_vg.fit(self.optimizer.space.transform(Xi_vg.tolist()), yi_vg)

                x1 = np.linspace(float(b1[0]), float(b1[1]), 150)
                x2 = np.linspace(float(b2[0]), float(b2[1]), 150)
                X1, X2 = np.meshgrid(x1, x2)
                grid_pts = np.c_[X1.ravel(), X2.ravel()]
                grid_trans = self.optimizer.space.transform(grid_pts.tolist())

                mu_vg = reg_vg.predict(grid_trans).reshape(X1.shape)
            except Exception:
                mu_vg = None

            # Colorbar limits based on mean +/- std deviation of group velocity
            mean_vg = float(np.mean(yi_vg))
            std_vg = float(np.std(yi_vg))

            vmin = max(0.0, mean_vg - std_vg)
            vmax = mean_vg + std_vg
            if vmax <= vmin:
                vmin = float(np.min(yi_vg))
                vmax = float(np.max(yi_vg))
                if vmax <= vmin:
                    vmax = vmin + 1e-4

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # Plot 1: Evaluated Group Velocity Scatter
            sc = ax1.scatter(
                Xi_vg[:, 0], Xi_vg[:, 1], c=yi_vg, cmap="viridis", norm=norm, s=40, edgecolors="black", linewidths=0.5, zorder=4
            )
            ax1.set_xlabel(f"${p1_name}/a$", fontsize=11)
            ax1.set_ylabel(f"${p2_name}/a$", fontsize=11)
            ax1.set_title(f"(a) Top Band #{top_b_num} $v_g$ at $\Delta k={delta_k}$ ($c$)", fontsize=12, fontweight="bold")
            ax1.set_xlim(float(b1[0]), float(b1[1]))
            ax1.set_ylim(float(b2[0]), float(b2[1]))
            ax1.set_aspect("equal", adjustable="box")
            ax1.grid(alpha=0.4, linestyle="--")

            # Continuous smooth contour levels
            cont_levels = np.linspace(vmin, vmax, 256)

            # Plot 2: GP Surrogate Map of Group Velocity
            if mu_vg is not None:
                heatmap = ax2.contourf(X1, X2, mu_vg, levels=cont_levels, cmap="viridis", norm=norm, extend="both")
                ax2.scatter(Xi_vg[:, 0], Xi_vg[:, 1], c="white", edgecolors="black", s=25, alpha=0.7, label="Evaluated points", zorder=5)
            else:
                heatmap = sc

            ax2.set_xlabel(f"${p1_name}/a$", fontsize=11)
            ax2.set_ylabel(f"${p2_name}/a$", fontsize=11)
            ax2.set_title(f"(b) GP Predicted $v_g$ Surface ($c$)", fontsize=12, fontweight="bold")
            ax2.set_xlim(float(b1[0]), float(b1[1]))
            ax2.set_ylim(float(b2[0]), float(b2[1]))
            ax2.set_aspect("equal", adjustable="box")
            ax2.grid(alpha=0.4, linestyle="--")

            for ax in (ax1, ax2):
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            cbar = fig.colorbar(heatmap, ax=[ax1, ax2], fraction=0.035, pad=0.04, extend="both")
            cbar.set_label(r"$v_g$ ($c$)", fontsize=13)
            cbar.ax.tick_params(labelsize=9)

            plt.savefig(fig_file, dpi=200, bbox_inches="tight")
            plt.close(fig)
            if verbose:
                print(f"Saved group velocity map plot to '{fig_file}'")

    def _get_script_path(self) -> Path:
        script_cfg = self.sim_cfg.get("ctl_script", "example.ctl")
        path = Path(script_cfg)
        if path.is_file():
            return path.resolve()
        w_path = (self.work_dir / script_cfg).resolve()
        if w_path.is_file():
            return w_path
        return Path(script_cfg)

    def _run_validation(self, best_params: Dict[str, float]) -> None:
        print("\nExecuting final validation simulation across FULL k-path with optimal parameters...")
        combined = {**self.fixed_params, **best_params}
        combined["display_symmetry?"] = "true"

        script_path = self._get_script_path()

        res = run_hpc(
            script=script_path,
            mpb_command_line_params=combined,
            use_mpi=True,
            cores=self.sim_cfg.get("cores", 4),
            wd=self.output_dir,
            auto_extract=True,
            auto_plot=True,
            only_gamma=False,  # Run full k-path
        )

        if res.returncode == 0:
            print(f"Validation complete! Optimal figures saved inside '{self.output_dir}'")
        else:
            print("Validation run finished with non-zero exit code.")


def run_bo(config_path: Union[str, os.PathLike], work_dir: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
    """
    Main programmatic API function to run Bayesian Optimization.
    """
    config = load_bo_config(config_path)
    if work_dir:
        config["simulation"]["work_dir"] = str(work_dir)

    opt = BayesianOptimizer(config)
    return opt.run()


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization for Photonic Crystal NZI / Dirac Cone Search."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="bo_config.yaml",
        help="Path to optimization YAML/JSON config file (default: bo_config.yaml)",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=None,
        help="Target working directory (overrides config work_dir)",
    )

    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_file():
        # Check in ctl directory if not found in cwd
        alt_path = Path("ctl") / config_path
        if alt_path.is_file():
            config_path = str(alt_path)

    run_bo(config_path=config_path, work_dir=args.dir)


if __name__ == "__main__":
    main()
