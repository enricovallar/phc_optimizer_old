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
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf

class CwdConfigSearchPathPlugin(SearchPathPlugin):
    """
    Hydra plugin that automatically prepends the local ./configs directory of the
    current working directory to Hydra's search path when present.
    """
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        cwd_configs = Path.cwd() / "configs"
        if cwd_configs.is_dir():
            search_path.prepend(
                provider="phc_cwd",
                path=f"file://{cwd_configs.resolve()}",
            )

Plugins.instance().register(CwdConfigSearchPathPlugin)

from skopt import Optimizer
from skopt.space import Real, Integer

from phc_nzi.runner import run_hpc
from phc_nzi.extractor import extract_frequencies
from phc_nzi.symmetry import analyze_symmetries_from_log
from phc_nzi.plotter import plot_band_structure, plot_epsilon
from phc_nzi.geometry_check import check_slab_connectivity

warnings.filterwarnings("ignore", category=UserWarning, module="skopt")



def validate_and_normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes an optimization configuration dictionary with defaults.
    """
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration dictionary. Expected a dict, got {type(config)}.")

    # 1. Simulation configuration
    sim_raw = config.get("simulation", {})
    sim = {
        "ctl_script": str(sim_raw.get("ctl_script", "main.ctl")),
        "work_dir": str(sim_raw.get("work_dir", "work")),
        "output_dir": str(sim_raw.get("output_dir", "bo_output")),
        "cores": int(sim_raw.get("cores", 4)),
        "parallel_workers": int(sim_raw.get("parallel_workers", sim_raw.get("local_workers", 4))),
        "only_gamma": bool(sim_raw.get("only_gamma", True)),
        "debug_timing": bool(sim_raw.get("debug_timing", True)),
    }
    config["simulation"] = sim

    # 2. Parameters & Fixed Parameters
    params_raw = config.get("parameters", {})
    if isinstance(params_raw, dict) and "search" in params_raw:
        config["parameters"] = params_raw["search"]
        if "fixed" in params_raw:
            config["fixed_parameters"] = {**params_raw["fixed"], **config.get("fixed_parameters", {})}
    elif not params_raw:
        raise ValueError("No optimization parameters defined under 'parameters' in config file.")
    config.setdefault("fixed_parameters", {})

    # 3. Target physics & symmetry specifications
    target_raw = config.get("target", {})
    sym = target_raw.get("symmetry", {})
    conn = target_raw.get("connectivity", {})
    vg = target_raw.get("group_velocity", {})

    target = {
        "symmetry_group": str(sym.get("group", target_raw.get("symmetry_group", "C4v"))),
        "polarization": str(sym.get("polarization", target_raw.get("polarization", "te"))),
        "target_irreps": list(sym.get("target_irreps", target_raw.get("target_irreps", ["A_2", "E", "E"]))),
        "irrep_occurrences": list(sym.get("irrep_occurrences", target_raw.get("irrep_occurrences", [1, 1, 1]))),
        "min_band": int(sym.get("min_band", target_raw.get("min_band", 2))),
        "degeneracy_tol": float(sym.get("degeneracy_tol", target_raw.get("degeneracy_tol", 0.005))),
        "target_cost": float(sym.get("target_cost", target_raw.get("target_cost", 0.0025))),
        "bypass_irrep_identification": bool(sym.get("bypass_irrep_identification", target_raw.get("bypass_irrep_identification", False))),
        "mode_indices": sym.get("mode_indices", target_raw.get("mode_indices")),

        "check_slab_connectivity": bool(conn.get("enabled", target_raw.get("check_slab_connectivity", False))),
        "enforce_connectivity": bool(conn.get("enabled", target_raw.get("enforce_connectivity", False))),
        "epsilon_threshold": float(conn.get("epsilon_threshold", target_raw.get("epsilon_threshold", 1.1))),
        "min_neck_width_px": int(conn.get("min_neck_width_px", target_raw.get("min_neck_width_px", 4))),

        "compute_group_velocity": bool(vg.get("enabled", target_raw.get("compute_group_velocity", False))),
        "delta_k": float(vg.get("delta_k", target_raw.get("delta_k", 0.01))),
        "vg_optimal_only": bool(vg.get("optimal_only", target_raw.get("vg_optimal_only", True))),
    }
    target["symmetry"] = sym
    target["connectivity"] = conn
    target["group_velocity"] = vg
    config["target"] = target

    # 4. Bayesian Optimizer Settings
    opt_raw = config.get("optimizer") or config.get("optimization", {})
    opt_iter = opt_raw.get("iterations", {})
    opt_surr = opt_raw.get("surrogate", {})
    opt_acq = opt_raw.get("acquisition", {})
    opt_vis = opt_raw.get("visualization", {})

    opt = {
        "max_iterations": int(opt_iter.get("max_iterations", opt_raw.get("max_iterations", 20))),
        "batch_size": int(opt_iter.get("batch_size", opt_raw.get("batch_size", 4))),
        "initial_points": int(opt_iter.get("initial_points", opt_raw.get("initial_points", 8))),
        "initial_sampling": str(opt_iter.get("initial_sampling", opt_raw.get("initial_sampling", "sobol"))),

        "model": str(opt_surr.get("model", opt_raw.get("model", "GP"))),
        "objective_mode": str(opt_surr.get("objective_mode", opt_raw.get("objective_mode", "log"))),
        "strategy": str(opt_surr.get("strategy", opt_raw.get("strategy", "cl_min"))),
        "random_state": int(opt_surr.get("random_state", opt_raw.get("random_state", 42))),

        "acq_func": str(opt_acq.get("acq_func", opt_raw.get("acq_func", "LCB"))),
        "acq_func_kwargs": opt_acq.get("acq_func_kwargs", opt_raw.get("acq_func_kwargs", {"kappa": 3.5})),
        "acq_optimizer": str(opt_acq.get("optimizer", opt_raw.get("acq_optimizer", "sampling"))),
        "n_points": int(opt_acq.get("n_points", opt_raw.get("n_points", 1000))),
        "n_restarts_optimizer": int(opt_acq.get("n_restarts", opt_raw.get("n_restarts_optimizer", 1))),

        "save_surrogate_freq": int(opt_vis.get("save_surrogate_freq", opt_raw.get("save_surrogate_freq", 0))),
        "neglect_sigma": bool(opt_vis.get("neglect_sigma", opt_raw.get("neglect_sigma", False))),
        "surrogate_colorbar_limits": opt_vis.get("colorbar_limits", opt_raw.get("surrogate_colorbar_limits")),
    }
    opt["iterations"] = opt_iter
    opt["surrogate"] = opt_surr
    opt["acquisition"] = opt_acq
    opt["visualization"] = opt_vis
    config["optimizer"] = opt

    # 5. Postprocessing Pipeline
    post_raw = config.get("postprocessing", {})
    post_loc = post_raw.get("locus", {})
    post_ref = post_raw.get("refinement", {})
    post_vg = post_raw.get("group_velocity", {})
    post_out = post_raw.get("output", {})

    post = {
        "enabled": bool(post_raw.get("enabled", False)),
        "method": str(post_loc.get("method", post_raw.get("method", "skeleton_spline"))),
        "threshold_percentile": float(post_loc.get("threshold_percentile", post_raw.get("threshold_percentile", 90.0))),
        "min_locus_area_px": int(post_loc.get("min_locus_area_px", post_raw.get("min_locus_area_px", 25))),
        "max_loci": int(post_loc.get("max_loci", post_raw.get("max_loci", 1))),
        "smoothness": float(post_loc.get("smoothness", post_raw.get("smoothness", 0.001))),
        "spline_degree": int(post_loc.get("spline_degree", post_raw.get("spline_degree", 3))),
        "sample_points": int(post_loc.get("sample_points", post_raw.get("sample_points", 50))),

        "ensure_degeneracy": bool(post_ref.get("enabled", post_raw.get("ensure_degeneracy", False))),
        "degeneracy_tolerance": float(post_ref.get("tolerance", post_raw.get("degeneracy_tolerance", 1.0e-5))),
        "max_refine_steps": int(post_ref.get("max_steps", post_raw.get("max_refine_steps", 6))),
        "refine_method": str(post_ref.get("method", post_raw.get("refine_method", "normal"))),

        "compute_group_velocity": bool(post_vg.get("enabled", post_raw.get("compute_group_velocity", False))),
        "delta_k": float(post_vg.get("delta_k", target["delta_k"])),

        "compute_band_diagram": bool(
            (post_raw.get("band_diagram", {}).get("enabled") if isinstance(post_raw.get("band_diagram"), dict) else None)
            if post_raw.get("band_diagram") is not None
            else (
                post_raw.get("band_structure", {}).get("enabled", True)
                if isinstance(post_raw.get("band_structure"), dict)
                else post_raw.get("compute_band_diagram", True)
            )
        ),

        "export_csv": bool(post_out.get("export_csv", post_raw.get("export_csv", True))),
        "plot_overlay": bool(post_out.get("plot_overlay", post_raw.get("plot_overlay", True))),
        "plot_profiles": bool(post_out.get("plot_profiles", post_raw.get("plot_profiles", True))),
    }
    post["locus"] = post_loc
    post["refinement"] = post_ref
    post["group_velocity"] = post_vg
    post["band_diagram"] = post_raw.get("band_diagram") or post_raw.get("band_structure") or {"enabled": post["compute_band_diagram"]}
    post["output"] = post_out
    config["postprocessing"] = post

    return config


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
            config = yaml.safe_load(f)

    return validate_and_normalize_config(config)


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
        self.config = validate_and_normalize_config(copy.deepcopy(config))
        self.sim_cfg = self.config["simulation"]
        self.params_cfg = self.config["parameters"]
        self.fixed_params = self.config.get("fixed_parameters", {})
        self.target_cfg = self.config["target"]
        self.opt_cfg = self.config["optimizer"]
        self.post_cfg = self.config.get("postprocessing", {})

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
        n_pts = self.opt_cfg.get("n_points", 1000)
        n_restarts = self.opt_cfg.get("n_restarts_optimizer", 3)

        acq_opt_kwargs = dict(self.opt_cfg.get("acq_optimizer_kwargs", {}))
        acq_opt_kwargs.setdefault("n_points", n_pts)
        acq_opt_kwargs.setdefault("n_restarts_optimizer", n_restarts)
        acq_opt_kwargs.setdefault("n_jobs", n_jobs)

        self.optimizer = Optimizer(
            dimensions=self.dimensions,
            base_estimator=self.opt_cfg.get("model", "GP"),
            n_initial_points=self.opt_cfg.get("initial_points", 8),
            initial_point_generator=self.opt_cfg.get("initial_sampling", "sobol"),
            acq_func=self.opt_cfg.get("acq_func", "LCB"),
            acq_func_kwargs=self.opt_cfg.get("acq_func_kwargs", {"kappa": 3.5}),
            acq_optimizer=self.opt_cfg.get("acq_optimizer", "auto"),
            acq_optimizer_kwargs=acq_opt_kwargs,
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
        pol = self.target_cfg.get("polarization", "te").lower()
        combined_params[f"run-{pol}?"] = "true"

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
                # Fallback for 2D vs 3D parity aliases (zeven <-> te, zodd <-> tm)
                alias_map = {
                    "zeven": "te",
                    "zodd": "tm",
                    "te": "zeven",
                    "tm": "zodd",
                }
                alt_pol = alias_map.get(pol)
                if alt_pol and f"{alt_pol}freqs" in extracted_data and "headers" in extracted_data[f"{alt_pol}freqs"]:
                    pol_key = f"{alt_pol}freqs"

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

            # Optional 2nd run: Group Velocity at small delta_k from Gamma for target bands
            vg_top_band = None
            compute_vg = bool(
                self.target_cfg.get("compute_group_velocity", False)
                or self.target_cfg.get("calculate_group_velocity", False)
            )

            # Auto-detect if point belongs to the purple region (low cost valley / high FOM)
            vg_optimal_only = bool(self.target_cfg.get("vg_optimal_only", True))
            min_cost_so_far = min(
                (float(r["raw_cost"]) for r in self.records if r.get("raw_cost") is not None and float(r["raw_cost"]) < 1.0),
                default=0.01
            )
            purple_threshold = max(min_cost_so_far * 3.0, 0.01)
            if "vg_cost_threshold" in self.target_cfg:
                purple_threshold = float(self.target_cfg["vg_cost_threshold"])

            should_run_vg = compute_vg and target_bands and (not vg_optimal_only or normalized_cost <= purple_threshold)

            if should_run_vg:
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

                        # Take the MORE POSITIVE group velocity among all target_bands
                        target_vgs = []
                        for rec_v in flat_recs:
                            if rec_v.get("parity", "").lower() == pol and int(rec_v.get("band", 0)) in target_bands:
                                vx_val = float(rec_v.get("vx", 0.0))
                                if np.isnan(vx_val):
                                    vx_val = float(rec_v.get("vg_mag", 0.0))
                                target_vgs.append(vx_val)

                        if target_vgs:
                            vg_top_band = float(max(target_vgs))

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


    def run(self) -> Dict[str, Any]:
        """
        Executes the main Bayesian Optimization loop.
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

        initial_generations = math.ceil(n_initial_pts / batch_size)
        total_generations = initial_generations + bo_generations

        print("==================================================================")
        print("Starting Photonic Crystal Bayesian Optimization")
        print(f"Working Directory:   '{self.work_dir}'")
        print(f"Output Directory:    '{self.output_dir}'")
        print(f"Parameters:          {self.param_names}")
        print(f"Target Irreps:       {self.target_cfg.get('target_irreps')} ({self.target_cfg.get('symmetry_group')})")
        print(f"Initial Sampling:    {self.opt_cfg.get('initial_sampling', 'sobol')} ({n_initial_pts} points = {initial_generations} initial gens)")
        if bo_generations > 0:
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

            init_method = self.opt_cfg.get("initial_sampling", "sobol").capitalize()
            if gen < initial_generations:
                phase_str = f"Initial {init_method} ({current_evals}/{n_initial_pts} pts)"
            else:
                bo_idx = gen - initial_generations + 1
                phase_str = f"Guided GP ({bo_idx}/{bo_generations})"

            pbar.set_postfix({"phase": phase_str, "best_cost": f"{best_score:.6f}"})
            pbar.update(1)

            if surrogate_freq > 0 and (gen + 1) % surrogate_freq == 0:
                self._plot_surrogate_map(final=False, verbose=False)
                self._plot_convergence(verbose=False)

        pbar.close()

        # Save optimizer model checkpoint
        joblib.dump(self.optimizer, self.model_file)

        best_idx = int(np.argmin(self.optimizer.yi))
        best_params = dict(zip(self.param_names, [float(x) for x in self.optimizer.Xi[best_idx]]))
        best_record = min(self.records, key=lambda r: r.get("raw_cost", float("inf"))) if self.records else {}

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

        # Plot convergence curve and surrogate map (which extracts and evaluates loci in locus_XX/pt_YY/)
        self._plot_convergence(verbose=True)
        self._plot_surrogate_map(final=True, verbose=True)

        # Fallback single validation run only if postprocessing was disabled or produced no loci
        post_cfg = getattr(self, "post_cfg", {}) or {}
        locus_dirs = list(self.output_dir.glob("locus_*"))
        if not post_cfg.get("enabled", False) or not locus_dirs:
            self._run_validation(best_params)

        return best_summary


    def _plot_convergence(self, verbose: bool = True) -> None:
        if not self.records:
            return
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.ticker import FormatStrFormatter

        costs = [r["raw_cost"] for r in self.records]
        iters_list = [r.get("generation", 1) for r in self.records]
        eval_nums = [r.get("eval_number", i + 1) for i, r in enumerate(self.records)]

        costs = np.array(costs)
        foms = 1.0 / np.maximum(costs, 1e-12)
        iterations = np.array(iters_list)
        eval_indices = np.array(eval_nums)
        n_evals = len(costs)

        n_initial = self.opt_cfg.get("initial_points", 8)
        n_initial_evals = min(n_initial, n_evals)

        unique_iters = sorted(list(set(iterations)))
        min_iter, max_iter = (min(unique_iters), max(unique_iters)) if unique_iters else (1, 1)

        # Initial sampling phase limit (iterations of initial points)
        initial_iters = sorted(list(set(iterations[eval_indices <= n_initial_evals]))) if n_initial_evals > 0 else [1]
        n_initial_iters = max(initial_iters) if initial_iters else 1
        div_point_iter = n_initial_iters + 0.5

        # Stack colors: single uniform color for initial sampling + winter colormap for active BO
        cmap_spring = plt.get_cmap("spring")
        cmap_winter = plt.get_cmap("winter")

        init_iters_list = [it for it in unique_iters if it <= n_initial_iters]
        bo_iters_list = [it for it in unique_iters if it > n_initial_iters]

        n_init = len(init_iters_list)
        n_bo = len(bo_iters_list)

        single_init_color = cmap_spring(0.4)
        init_colors = [single_init_color] * n_init

        if n_bo == 1:
            bo_colors = [cmap_winter(0.5)]
        elif n_bo > 1:
            bo_colors = [cmap_winter(v) for v in np.linspace(0.15, 0.85, n_bo)]
        else:
            bo_colors = []

        stacked_colors = init_colors + bo_colors
        stacked_cmap = mcolors.ListedColormap(stacked_colors)
        bounds_iter = np.arange(min_iter - 0.5, max_iter + 1.5, 1)
        discrete_norm = mcolors.BoundaryNorm(bounds_iter, len(stacked_colors))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.0))

        # ---------------------------------------------------------
        # Plot 1 (Left / (a)): Evaluated Points by Iteration Map
        # ---------------------------------------------------------
        if len(self.param_names) == 2 and hasattr(self.optimizer, "Xi") and len(self.optimizer.Xi) > 0:
            p1_name, p2_name = self.param_names[0], self.param_names[1]
            b1 = self.param_bounds[0]
            b2 = self.param_bounds[1]

            Xi = np.array(self.optimizer.Xi)
            yi = np.array(self.optimizer.yi)

            # Match records to Xi
            matched_iters = []
            for idx, x_pt in enumerate(Xi):
                mit = 1
                for r in self.records:
                    r_pt = [r["params"][name] for name in self.param_names if name in r["params"]]
                    if len(r_pt) == len(x_pt) and np.allclose(r_pt, x_pt, atol=1e-5):
                        mit = r.get("generation", 1)
                        break
                matched_iters.append(mit)
            matched_iters = np.array(matched_iters)

            best_idx = int(np.argmin(yi))
            best_x = Xi[best_idx]

            p1_label = f"${p1_name[0]}_{{{p1_name[1:]}}}/a$" if len(p1_name) > 1 and p1_name[1:].isdigit() else f"${p1_name}/a$"
            p2_label = f"${p2_name[0]}_{{{p2_name[1:]}}}/a$" if len(p2_name) > 1 and p2_name[1:].isdigit() else f"${p2_name}/a$"

            sc1 = ax1.scatter(
                Xi[:, 0],
                Xi[:, 1],
                c=matched_iters,
                cmap=stacked_cmap,
                norm=discrete_norm,
                s=42,
                edgecolors="black",
                linewidths=0.5,
                marker="o",
                zorder=4,
                label="Explored points",
            )
            ax1.scatter(
                best_x[0],
                best_x[1],
                c="gold",
                edgecolors="black",
                marker="*",
                s=220,
                zorder=6,
                label="Optimal Point",
            )

            ax1.set_xlabel(p1_label, fontsize=11)
            ax1.set_ylabel(p2_label, fontsize=11)
            ax1.set_title("(a) Evaluated Points by Iteration", fontsize=12, fontweight="bold")
            ax1.set_xlim(float(b1[0]), float(b1[1]))
            ax1.set_ylim(float(b2[0]), float(b2[1]))
            ax1.set_aspect("equal", adjustable="box")
            ax1.grid(alpha=0.4, linestyle="--")
            ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.1)
            cbar1 = fig.colorbar(sc1, cax=cax1, ticks=np.arange(min_iter, max_iter + 1))
            cbar1.set_label("Iteration #", fontsize=11)
            cbar1.ax.tick_params(labelsize=9)

            ax1.legend(loc="upper right", fontsize=9, facecolor="white", edgecolor="black", framealpha=0.9)
        else:
            best_so_far = np.maximum.accumulate(foms)
            ax1.plot(eval_indices, foms, "-", color="gray", alpha=0.25)
            ax1.plot(eval_indices, best_so_far, "r-", linewidth=2.5, zorder=5, label=r"Best $\mathrm{E}[C]^{-1}$ So Far")
            ax1.set_yscale("log")
            ax1.set_xlabel("Evaluation #", fontsize=11)
            ax1.set_ylabel(r"$\mathrm{E}[C]^{-1}$", fontsize=13)
            ax1.set_title("(a) Convergence vs. Evaluation #", fontsize=12, fontweight="bold")
            ax1.grid(True, which="both", ls="--", alpha=0.4)
            ax1.legend(loc="upper left")

        # ---------------------------------------------------------
        # Plot 2 (Right / (b)): E[C]^-1 Distribution vs. Iteration #
        # ---------------------------------------------------------
        for i, it in enumerate(unique_iters):
            mask = (iterations == it)
            c_it = stacked_colors[i]
            it_foms = foms[mask]
            ax2.scatter([it] * len(it_foms), it_foms, color=c_it, s=40, edgecolors="black", linewidths=0.4, zorder=4, alpha=0.85)

        # Background regions with subtle initial and winter tints
        ax2.axvspan(min_iter - 0.5, div_point_iter, color=single_init_color, alpha=0.10, label="Initial Phase")
        ax2.axvspan(div_point_iter, max_iter + 0.5, color=cmap_winter(0.5), alpha=0.10, label="BO Phase (Winter)")
        ax2.axvline(x=div_point_iter, color="gray", linestyle="--", alpha=0.8, label="BO Phase Start")

        ax2.set_yscale("log")
        ax2.set_xlabel("Iteration #", fontsize=11)
        ax2.set_ylabel(r"$\mathrm{E}[C]^{-1}$", fontsize=13)
        ax2.set_title(r"(b) $\mathrm{E}[C]^{-1}$ Distribution vs. Iteration #", fontsize=12, fontweight="bold")
        ax2.set_xticks(np.arange(min_iter, max_iter + 1))
        ax2.set_xlim(min_iter - 0.5, max_iter + 0.5)
        ax2.grid(True, which="both", ls="--", alpha=0.4)
        ax2.legend(loc="upper left", fontsize=9, facecolor="white", edgecolor="black")

        plt.tight_layout()
        conv_file = self.output_dir / "bo_convergence.png"
        fig.savefig(conv_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"Saved convergence plot to '{conv_file}'")

    def _plot_surrogate_map(self, final: bool = False, verbose: bool = True) -> None:
        """
        Generates and saves a 2-panel figure matching pillars_c6v_1.ipynb styling:
        - Plot 1 (Left): Raw optimization evaluated points colored by FOM (E[C]^-1) using LogNorm and 'cool' colormap.
        - Plot 2 (Right): GP Surrogate model predicted FOM (E[C]^-1) contour landscape with LogNorm and 'cool' colormap,
          overlaid with sampled points and optimal point star.
        """
        import matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        from matplotlib.ticker import FormatStrFormatter, LogFormatterSciNotation, LogLocator

        if not hasattr(self.optimizer, "Xi") or len(self.optimizer.Xi) == 0:
            return

        model = self.optimizer.models[-1] if (hasattr(self.optimizer, "models") and self.optimizer.models) else None
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

            mode = self.opt_cfg.get("objective_mode", "log").lower()

            # Predict surrogate landscape
            # When evaluated records are available, fit clean GP on valid physical points (cost < 0.5)
            # to eliminate boundary penalty step distortions from the smooth dispersion landscape
            clean_pts = []
            clean_y = []
            for r in self.records:
                if "params" in r and "raw_cost" in r:
                    c_val = float(r["raw_cost"])
                    if c_val < 0.5:
                        clean_pts.append([float(r["params"][p1_name]), float(r["params"][p2_name])])
                        clean_y.append(np.log10(max(c_val, 1e-12)) if mode == "log" else c_val)

            fitted_clean = False
            if len(clean_pts) >= 10:
                try:
                    from skopt.learning import GaussianProcessRegressor
                    from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel
                    clean_gp = GaussianProcessRegressor(
                        kernel=Matern(length_scale=0.035, length_scale_bounds=(0.015, 0.08), nu=2.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1)),
                        normalize_y=True,
                        n_restarts_optimizer=3,
                        random_state=42,
                    )
                    clean_gp.fit(np.array(clean_pts), np.array(clean_y))
                    mu_grid = clean_gp.predict(grid_points).reshape(X1.shape)
                    std_grid = np.zeros_like(mu_grid)
                    fitted_clean = True
                except Exception:
                    fitted_clean = False

            if not fitted_clean:
                if hasattr(model, "predict"):
                    try:
                        if hasattr(model, "return_std"):
                            mu_grid, std_grid = model.predict(grid_points_transformed, return_std=True)
                        else:
                            mu_grid, std_grid = model.predict(grid_points_transformed, return_std=True)
                    except Exception:
                        try:
                            mu_grid = model.predict(grid_points_transformed)
                            std_grid = np.zeros_like(mu_grid)
                        except Exception:
                            mu_grid = np.zeros(len(grid_points))
                            std_grid = np.zeros_like(mu_grid)
                    mu_grid = mu_grid.reshape(X1.shape)
                    std_grid = std_grid.reshape(X1.shape)
                else:
                    mu_grid = np.zeros(X1.shape)
                    std_grid = np.zeros(X1.shape)

            mode = self.opt_cfg.get("objective_mode", "log").lower()
            Xi = np.array(self.optimizer.Xi)
            yi = np.array(self.optimizer.yi)

            # Build fast lookup dictionary for evaluated records by parameter coordinates
            rec_map = {}
            for r in self.records:
                if "params" in r and "raw_cost" in r:
                    key = tuple(round(float(r["params"].get(k, 0.0)), 6) for k in self.param_names)
                    rec_map[key] = float(r["raw_cost"])

            # Map raw cost values to original physical scale matching each Xi point coordinates
            raw_costs_list = []
            for idx, x_pt in enumerate(Xi):
                key = tuple(round(float(v), 6) for v in x_pt)
                if key in rec_map and not np.isnan(rec_map[key]):
                    raw_costs_list.append(rec_map[key])
                elif mode == "log":
                    raw_costs_list.append(10 ** float(yi[idx]))
                else:
                    raw_costs_list.append(float(yi[idx]))

            raw_costs = np.array(raw_costs_list)

            # Compute Expectation Value of Cost E[C] (or deterministic mean if neglect_sigma is True)
            include_sigma = not bool(
                self.opt_cfg.get("neglect_sigma", False)
                or self.opt_cfg.get("surrogate_neglect_sigma", False)
                or (self.opt_cfg.get("surrogate_include_std") is False)
                or (self.opt_cfg.get("include_uncertainty") is False)
            )

            if mode == "log":
                if include_sigma:
                    # For y = log10(C), C is Log-Normal: E[C] = 10^(mu + (ln(10)/2) * std^2)
                    exp_c_grid = 10 ** (mu_grid + (np.log(10) / 2.0) * (std_grid ** 2))
                else:
                    # Neglect posterior uncertainty: deterministic mean C = 10^mu
                    exp_c_grid = 10 ** mu_grid
            else:
                exp_c_grid = mu_grid

            fom_raw = 1.0 / np.maximum(raw_costs, 1e-12)
            predicted_e_c_inv = 1.0 / np.maximum(exp_c_grid, 1e-12)

            # Compute shared LogNorm scale bounds using explicit config or robust percentiles
            user_limits = (
                self.opt_cfg.get("surrogate_colorbar_limits")
                or self.opt_cfg.get("colorbar_limits")
                or self.opt_cfg.get("plot_limits")
            )
            if user_limits and len(user_limits) == 2:
                vmin, vmax = float(user_limits[0]), float(user_limits[1])
            else:
                all_fom_vals = np.r_[fom_raw, predicted_e_c_inv.ravel()]
                finite_pos = all_fom_vals[np.isfinite(all_fom_vals) & (all_fom_vals > 0)]
                p_lims = self.opt_cfg.get("surrogate_percentiles", [1, 99])
                p_low, p_high = float(p_lims[0]), float(p_lims[1])

                if finite_pos.size > 0:
                    vmin = max(float(np.percentile(finite_pos, p_low)), 1e-12)
                    vmax = float(np.percentile(finite_pos, p_high))
                    if vmax <= vmin:
                        vmax = vmin * 10
                else:
                    vmin, vmax = 1.0, 1000.0

            log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

            target_irreps = self.target_cfg.get("target_irreps", ["A_2", "E", "E"])
            target_str = " - ".join(dict.fromkeys(target_irreps))

            fig = plt.figure(figsize=(13.0, 5.8))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.25)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            cax = fig.add_subplot(gs[0, 2])

            p1_label = f"${p1_name[0]}_{{{p1_name[1:]}}}/a$" if len(p1_name) > 1 and p1_name[1:].isdigit() else f"${p1_name}/a$"
            p2_label = f"${p2_name[0]}_{{{p2_name[1:]}}}/a$" if len(p2_name) > 1 and p2_name[1:].isdigit() else f"${p2_name}/a$"

            # ---------------------------------------------------------
            # Plot 1 (Left): Raw Optimization Data
            # ---------------------------------------------------------
            sc_raw = ax1.scatter(
                Xi[:, 0], Xi[:, 1], c=fom_raw, cmap="cool", norm=log_norm, s=35, edgecolors="black", linewidths=0.5, marker="o", zorder=4
            )

            best_idx = int(np.argmin(yi))
            best_x = Xi[best_idx]
            ax1.scatter(best_x[0], best_x[1], c="gold", edgecolors="black", marker="*", s=220, zorder=6, label="Optimal Point")

            ax1.set_xlabel(p1_label, fontsize=11)
            ax1.set_ylabel(p2_label, fontsize=11)
            ax1.set_title(f"(a) Evaluated Points ({target_str})", fontsize=12, fontweight="bold")
            ax1.set_xlim(float(b1[0]), float(b1[1]))
            ax1.set_ylim(float(b2[0]), float(b2[1]))
            ax1.set_aspect("equal", adjustable="box")
            ax1.grid(alpha=0.4, linestyle="--")

            # ---------------------------------------------------------
            # Plot 2 (Right): GP Surrogate Expectation Map E[C]^-1
            # ---------------------------------------------------------
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 200)
            heatmap = ax2.contourf(
                X1, X2, predicted_e_c_inv, levels=levels, cmap="cool", norm=log_norm, extend="both"
            )

            # ---------------------------------------------------------
            # Postprocessing: Parametric Optimal Loci Extraction & Evaluation
            # ---------------------------------------------------------
            post_cfg = getattr(self, "post_cfg", {}) or {}
            loci = None
            if post_cfg.get("enabled", False):
                try:
                    from .locus import extract_optimal_loci, export_loci_to_json
                    loci = extract_optimal_loci(x1, x2, predicted_e_c_inv, post_cfg)
                    if loci and post_cfg.get("plot_overlay", True):
                        for locus in loci:
                            l_id = locus["locus_id"]
                            r1_pts = locus["r1"]
                            r2_pts = locus["r2"]
                            lbl_gp = "GP Locus" if len(loci) == 1 else f"GP Locus #{l_id}"
                            ax2.plot(
                                r1_pts,
                                r2_pts,
                                color="#00FF66",
                                linestyle="--",
                                linewidth=2.0,
                                alpha=0.85,
                                label=lbl_gp,
                                zorder=7,
                            )
                except Exception as e:
                    if verbose:
                        print(f"Note: Postprocessing locus extraction notice: {e}")

            ax2.scatter(Xi[:, 0], Xi[:, 1], c="white", edgecolors="black", s=25, alpha=0.7, label="Explored points", zorder=5)
            ax2.scatter(best_x[0], best_x[1], c="gold", edgecolors="black", marker="*", s=220, zorder=6, label="Optimal Point")

            ax2.set_xlabel(p1_label, fontsize=11)
            ax2.set_ylabel(p2_label, fontsize=11)
            ax2.set_title(f"(b) {target_str} Tracking", fontsize=12, fontweight="bold")
            ax2.set_xlim(float(b1[0]), float(b1[1]))
            ax2.set_ylim(float(b2[0]), float(b2[1]))
            ax2.set_aspect("equal", adjustable="box")
            ax2.grid(alpha=0.4, linestyle="--")

            # Shared tick format
            for ax in (ax1, ax2):
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # Dedicated colorbar axis with inward ticks matching snippet
            cbar = fig.colorbar(heatmap, cax=cax, extend="both", format=LogFormatterSciNotation())
            cbar.set_label(r"$\mathbb{E}[C(r_1, r_2)]^{-1}$" if len(self.param_names) == 2 else r"$\mathbb{E}[C]^{-1}$", fontsize=13)
            cbar.locator = LogLocator(base=10.0, numticks=6)
            cbar.update_ticks()
            cbar.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
            cbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())
            cbar.ax.tick_params(which="major", direction="in", length=5)
            cbar.ax.tick_params(which="minor", direction="in", length=2.5)

            # Update legend and save initial figure with GP-locus immediately
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels1 + labels2, handles1 + handles2))
            leg = fig.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, facecolor="white", edgecolor="black")

            fig.savefig(surrogate_file, dpi=200, bbox_inches="tight")
            if verbose:
                print(f"Saved surrogate map plot (with GP locus) to '{surrogate_file}'")

            # Run refinement & simulations on final pass
            if final and post_cfg.get("enabled", False) and loci:
                try:
                    from .locus import export_loci_to_json
                    loci = self._evaluate_locus_simulations(loci, verbose=verbose)

                    ref_enabled = bool(
                        post_cfg.get("refinement", {}).get("enabled", post_cfg.get("ensure_degeneracy", False))
                    )
                    has_refined_data = any(l.get("r1_unrefined") is not None for l in loci)

                    if post_cfg.get("plot_overlay", True) and ref_enabled and has_refined_data:
                        for locus in loci:
                            l_id = locus["locus_id"]
                            r1_ref = locus["r1"]
                            r2_ref = locus["r2"]
                            lbl_ref = "Refined Locus" if len(loci) == 1 else f"Refined Locus #{l_id}"
                            ax2.plot(
                                r1_ref,
                                r2_ref,
                                color="#00FFFF",
                                linestyle="-",
                                linewidth=2.2,
                                alpha=0.95,
                                label=lbl_ref,
                                zorder=8,
                            )

                        # Update legend and re-save figure with refined locus
                        handles1, labels1 = ax1.get_legend_handles_labels()
                        handles2, labels2 = ax2.get_legend_handles_labels()
                        by_label = dict(zip(labels1 + labels2, handles1 + handles2))
                        leg.remove()
                        fig.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, facecolor="white", edgecolor="black")

                        fig.savefig(surrogate_file, dpi=200, bbox_inches="tight")
                        if verbose:
                            print(f"Updated surrogate map plot (with refined locus) to '{surrogate_file}'")

                    json_path = self.output_dir / "bo_loci.json"
                    export_loci_to_json(loci, json_path)
                except Exception as e:
                    if verbose:
                        print(f"Note: Postprocessing refinement overlay notice: {e}")

            plt.close(fig)

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





    def _evaluate_group_velocity_at_point(
        self, param_dict: Dict[str, float], target_bands: List[int]
    ) -> float:
        """
        Runs a dedicated MPB simulation at k = (delta_k, 0, 0) for a specific parameter point
        (e.g., GP surrogate peak), extracting the MOST POSITIVE group velocity among target_bands.
        """
        pol = self.target_cfg.get("polarization", "te").lower()
        delta_k = float(self.target_cfg.get("delta_k", 0.01))

        combined_params = {**self.fixed_params, **param_dict}
        combined_params["display_symmetry?"] = "false"
        combined_params["only_gamma?"] = "false"
        combined_params["display_group_velocity?"] = "true"
        combined_params["delta_k"] = delta_k
        combined_params["delta_k_mode?"] = "true"

        with tempfile.TemporaryDirectory() as vg_temp_dir:
            res_vg = run_hpc(
                script=self._get_script_path(),
                mpb_command_line_params=combined_params,
                use_mpi=False,
                cores=self.sim_cfg.get("cores", 4),
                wd=vg_temp_dir,
                auto_extract=True,
                auto_plot=False,
                only_gamma=False,
                verbose=False,
            )
            vg_log = Path(vg_temp_dir) / "output" / "output.out"
            if vg_log.is_file():
                from phc_nzi.extractor import extract_group_velocities
                vg_data = extract_group_velocities(output_path=vg_log, save_data=False, verbose=False)
                flat_recs = vg_data.get("flat_records", [])

                target_vgs = []
                for rec_v in flat_recs:
                    if rec_v.get("parity", "").lower() == pol and int(rec_v.get("band", 0)) in target_bands:
                        vx_val = float(rec_v.get("vx", 0.0))
                        if np.isnan(vx_val):
                            vx_val = float(rec_v.get("vg_mag", 0.0))
                        target_vgs.append(vx_val)

                if target_vgs:
                    return float(max(target_vgs))

        return 0.0

    def _evaluate_point_gamma_gap(
        self, p_dict: Dict[str, float], target_bands: Optional[List[int]] = None
    ) -> Tuple[float, float, List[int], Dict[str, Any]]:
        """
        Fast evaluation of the signed frequency gap at Gamma point (k=0).
        Runs MPB in only_gamma mode (0.05s) and computes:
          - signed_gap: frequency difference between singlet and doublet mode
          - normalized_cost: |Delta omega| / omega_0
          - target_bands: identified band indices
          - full_map: irrep mapping dict
        """
        p_vals = [float(p_dict[k]) for k in self.param_names if k in p_dict]
        cost, freq_dirac, label, full_map, t_bands, corrections, conn_status, _, _ = self._evaluate_single(
            (0, p_vals)
        )

        bands = t_bands or target_bands or [4, 5, 6]
        signed_gap = float(cost)

        if full_map and bands:
            singlet_freqs = [full_map[b][2] for b in bands if b in full_map and full_map[b][0] in ("A_1", "A_2", "B_1", "B_2")]
            doublet_freqs = [full_map[b][2] for b in bands if b in full_map and full_map[b][0] in ("E", "E_1", "E_2")]
            if singlet_freqs and doublet_freqs:
                signed_gap = float(np.mean(doublet_freqs) - np.mean(singlet_freqs))
            else:
                b_max = max(bands)
                b_min = min(bands)
                if b_max in full_map and b_min in full_map:
                    signed_gap = float(full_map[b_max][2] - full_map[b_min][2])

        return signed_gap, float(cost), bands, full_map

    def _refine_single_point_degeneracy(
        self,
        p_dict: Dict[str, float],
        normal_vec: np.ndarray,
        tol: float = 1e-5,
        max_steps: int = 6,
        target_bands: Optional[List[int]] = None,
        method: str = "normal",
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Fine-tunes a sampled locus point to exact degeneracy (cost < tol)
        using 1D normal line search / secant root-finding with Gamma-only evaluations.
        """
        best_p = dict(p_dict)
        best_cost = float("inf")
        best_gap = float("inf")

        # Determine unit direction for 1D search
        if method == "r2" and len(self.param_names) >= 2:
            direction = np.array([0.0, 1.0])
        elif method == "r1" and len(self.param_names) >= 1:
            direction = np.array([1.0, 0.0])
        else:
            norm_len = float(np.hypot(normal_vec[0], normal_vec[1]))
            if norm_len > 1e-8:
                direction = np.array([float(normal_vec[0]) / norm_len, float(normal_vec[1]) / norm_len])
            else:
                direction = np.array([0.0, 1.0])

        p1_name = self.param_names[0] if len(self.param_names) >= 1 else "r1"
        p2_name = self.param_names[1] if len(self.param_names) >= 2 else "r2"

        def get_point_at(delta: float) -> Dict[str, float]:
            cur = dict(p_dict)
            cur[p1_name] = float(p_dict[p1_name] + delta * direction[0])
            if len(self.param_names) >= 2:
                cur[p2_name] = float(p_dict[p2_name] + delta * direction[1])
            if len(self.param_bounds) >= 1:
                cur[p1_name] = float(np.clip(cur[p1_name], self.param_bounds[0][0], self.param_bounds[0][1]))
            if len(self.param_bounds) >= 2:
                cur[p2_name] = float(np.clip(cur[p2_name], self.param_bounds[1][0], self.param_bounds[1][1]))
            return cur

        # Step 0: Evaluate at delta = 0
        gap0, cost0, bands, _ = self._evaluate_point_gamma_gap(get_point_at(0.0), target_bands)
        best_p = get_point_at(0.0)
        best_cost = cost0
        best_gap = gap0

        if cost0 < tol:
            return best_p, gap0, cost0

        step_mag = 0.002
        delta_max = 0.035

        # Probe in positive direction
        gap_pos, cost_pos, _, _ = self._evaluate_point_gamma_gap(get_point_at(step_mag), target_bands)
        if cost_pos < best_cost:
            best_p = get_point_at(step_mag)
            best_cost = cost_pos
            best_gap = gap_pos
        if cost_pos < tol:
            return best_p, gap_pos, cost_pos

        # Probe in negative direction if positive didn't improve or if we need a gradient
        if cost_pos >= cost0 or cost_pos >= 0.99:
            gap_neg, cost_neg, _, _ = self._evaluate_point_gamma_gap(get_point_at(-step_mag), target_bands)
            if cost_neg < best_cost:
                best_p = get_point_at(-step_mag)
                best_cost = cost_neg
                best_gap = gap_neg
            if cost_neg < tol:
                return best_p, gap_neg, cost_neg
            deltas = [0.0, -step_mag]
            gaps = [gap0, gap_neg]
            costs = [cost0, cost_neg]
        else:
            deltas = [0.0, step_mag]
            gaps = [gap0, gap_pos]
            costs = [cost0, cost_pos]

        max_iter = max(10, int(max_steps))
        for step in range(2, max_iter):
            d_prev, d_curr = deltas[-2], deltas[-1]
            g_prev, g_curr = gaps[-2], gaps[-1]
            c_curr = costs[-1]

            # If last step was invalid / disconnected (cost >= 0.99), bisect backwards
            if c_curr >= 0.99:
                d_next = 0.5 * (d_prev + d_curr)
            else:
                denom = g_curr - g_prev
                if abs(denom) < 1e-10:
                    d_next = d_curr + (step_mag * 0.5 if step % 2 == 0 else -step_mag * 0.5)
                else:
                    d_next = d_curr - g_curr * (d_curr - d_prev) / denom

            d_next = float(np.clip(d_next, -delta_max, delta_max))

            # Avoid repeated evaluation at same delta
            if any(abs(d_next - d) < 1e-5 for d in deltas):
                d_next = d_curr + (step_mag * 0.25 if g_curr > 0 else -step_mag * 0.25)
                d_next = float(np.clip(d_next, -delta_max, delta_max))

            pt_next = get_point_at(d_next)
            g_next, c_next, _, _ = self._evaluate_point_gamma_gap(pt_next, target_bands)

            deltas.append(d_next)
            gaps.append(g_next)
            costs.append(c_next)

            if c_next < best_cost:
                best_p = pt_next
                best_cost = c_next
                best_gap = g_next

            if c_next < tol:
                return best_p, g_next, c_next

        return best_p, best_gap, best_cost

    def _refine_locus_degeneracy(
        self, loci: List[Dict[str, Any]], verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes local degeneracy refinement for all sampled points across each extracted locus
        in parallel using fast Gamma-only evaluations.
        """
        from .locus import compute_curve_normals

        post_cfg = getattr(self, "post_cfg", {}) or {}
        ref_cfg = post_cfg.get("refinement", {}) if isinstance(post_cfg.get("refinement"), dict) else {}
        tol = float(ref_cfg.get("tolerance", post_cfg.get("degeneracy_tolerance", 1.0e-5)))
        max_steps = int(ref_cfg.get("max_steps", post_cfg.get("max_refine_steps", 10)))
        method = str(ref_cfg.get("method", post_cfg.get("refine_method", "normal"))).lower()
        exclude_unref = bool(ref_cfg.get("exclude_unrefined", post_cfg.get("exclude_unrefined", True)))
        max_residual_gap = float(ref_cfg.get("max_residual_gap", post_cfg.get("max_residual_gap", 1.0e-4)))

        worker_cand = (
            self.sim_cfg.get("parallel_workers")
            or self.opt_cfg.get("batch_size")
            or self.sim_cfg.get("cores")
            or 4
        )
        max_workers = min(max(1, int(worker_cand)), 24)

        for locus in loci:
            l_id = locus["locus_id"]
            r1_pts = np.asarray(locus["r1"], dtype=float)
            r2_pts = np.asarray(locus["r2"], dtype=float)
            n_pts = len(r1_pts)

            # Preserve unrefined GP surrogate coordinates
            locus["r1_unrefined"] = list(r1_pts)
            locus["r2_unrefined"] = list(r2_pts)

            normals = compute_curve_normals(r1_pts, r2_pts)

            if verbose:
                print(f"Refining degeneracy for Locus #{l_id} ({n_pts} points, tol={tol:.1e}, method='{method}', {max_workers} workers)...")

            tasks = []
            for i in range(n_pts):
                p_entry = {self.param_names[0]: float(r1_pts[i])}
                if len(self.param_names) >= 2:
                    p_entry[self.param_names[1]] = float(r2_pts[i])
                tasks.append((i, p_entry, normals[i]))

            def refine_worker(item):
                idx, p_entry, norm_vec = item
                refined_p, gap_val, cost_val = self._refine_single_point_degeneracy(
                    p_dict=p_entry,
                    normal_vec=norm_vec,
                    tol=tol,
                    max_steps=max_steps,
                    method=method,
                )
                return idx, refined_p, gap_val, cost_val

            results = []
            pbar = tqdm(total=n_pts, desc=f"Refining Locus #{l_id}", unit="pt", disable=not verbose)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(refine_worker, item): item[0] for item in tasks}
                for f in as_completed(futures):
                    res = f.result()
                    results.append(res)
                    idx, refined_p, gap_val, cost_val = res
                    pbar.set_postfix({"pt": f"{idx+1:02d}/{n_pts:02d}", "gap": f"{cost_val:.1e}"})
                    pbar.update(1)
            pbar.close()

            r1_ref = [0.0] * n_pts
            r2_ref = [0.0] * n_pts
            gaps_ref = [0.0] * n_pts
            costs_ref = [0.0] * n_pts

            for idx, refined_p, gap_val, cost_val in results:
                r1_ref[idx] = float(refined_p[self.param_names[0]])
                if len(self.param_names) >= 2:
                    r2_ref[idx] = float(refined_p[self.param_names[1]])
                gaps_ref[idx] = float(gap_val)
                costs_ref[idx] = float(cost_val)

            is_valid_list = []
            validity_status_list = []
            for idx in range(n_pts):
                gap_v = costs_ref[idx]
                if gap_v <= max_residual_gap:
                    is_valid_list.append(True)
                    validity_status_list.append(f"PASSED: Degeneracy achieved (gap = {gap_v:.2e} <= {max_residual_gap:.2e})")
                else:
                    is_valid_list.append(False)
                    validity_status_list.append(f"FAILED: Residual gap ({gap_v:.2e}) exceeds tolerance limit ({max_residual_gap:.2e})")

            locus["r1"] = r1_ref
            locus["r2"] = r2_ref
            locus["residual_gap"] = costs_ref
            locus["gaps"] = gaps_ref
            locus["is_valid"] = is_valid_list
            locus["validity_status"] = validity_status_list

            if exclude_unref:
                valid_indices = [i for i, v in enumerate(is_valid_list) if v]
                num_pruned = n_pts - len(valid_indices)
                if valid_indices and num_pruned > 0:
                    # If points were pruned from a closed cycle, rotate valid_indices
                    # so that the curve is ordered continuously along the remaining open arc
                    # rather than jumping across the pruned gap.
                    gap_cut = None
                    for k in range(len(valid_indices) - 1):
                        if valid_indices[k + 1] > valid_indices[k] + 1:
                            gap_cut = k + 1
                            break
                    if gap_cut is not None:
                        valid_indices = valid_indices[gap_cut:] + valid_indices[:gap_cut]
                    locus["is_closed"] = False

                    locus["r1"] = [r1_ref[i] for i in valid_indices]
                    locus["r2"] = [r2_ref[i] for i in valid_indices]
                    locus["residual_gap"] = [costs_ref[i] for i in valid_indices]
                    locus["gaps"] = [gaps_ref[i] for i in valid_indices]
                    locus["is_valid"] = [is_valid_list[i] for i in valid_indices]
                    locus["validity_status"] = [validity_status_list[i] for i in valid_indices]
                    if "fom" in locus and len(locus["fom"]) == n_pts:
                        locus["fom"] = [locus["fom"][i] for i in valid_indices]
                    if "r1_unrefined" in locus and len(locus["r1_unrefined"]) == n_pts:
                        locus["r1_unrefined"] = [locus["r1_unrefined"][i] for i in valid_indices]
                    if "r2_unrefined" in locus and len(locus["r2_unrefined"]) == n_pts:
                        locus["r2_unrefined"] = [locus["r2_unrefined"][i] for i in valid_indices]

                if verbose:
                    max_gap = max(locus["residual_gap"]) if locus["residual_gap"] else 0.0
                    mean_gap = sum(locus["residual_gap"]) / max(len(locus["residual_gap"]), 1)
                    if num_pruned > 0 and valid_indices:
                        print(
                            f"  Locus #{l_id} refined: {len(locus['r1'])}/{n_pts} points valid "
                            f"(pruned {num_pruned} points exceeding gap {max_residual_gap:.1e}). "
                            f"Max residual gap = {max_gap:.2e}, Mean = {mean_gap:.2e}"
                        )
                    elif num_pruned > 0 and not valid_indices:
                        print(
                            f"  Locus #{l_id} refined: All {n_pts} points retained (residual gaps {min(costs_ref):.2e} - {max(costs_ref):.2e} exceed cutoff {max_residual_gap:.1e}). "
                            f"Mean gap = {mean_gap:.2e}"
                        )
                    else:
                        print(f"  Locus #{l_id} refined: All {n_pts} points within tolerance. Max residual gap = {max_gap:.2e}, Mean = {mean_gap:.2e}")
            else:
                if verbose:
                    max_gap = max(costs_ref)
                    mean_gap = sum(costs_ref) / len(costs_ref)
                    print(f"  Locus #{l_id} refined: Max residual gap = {max_gap:.2e}, Mean = {mean_gap:.2e}")

        return loci

    def _evaluate_locus_simulations(
        self, loci: List[Dict[str, Any]], verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes full MPB simulations for every sampled point along each extracted locus
        in parallel, organizing all outputs into dedicated per-locus and per-point directories:
          <output_dir>/locus_01/
            bo_locus.csv
            bo_locus_profile.png
            pt_01/
              output.out, band_structure.png, epsilon.png, point_info.json, ...
            pt_02/
              ...
        """
        import shutil

        post_cfg = getattr(self, "post_cfg", {}) or {}
        ensure_deg = bool(post_cfg.get("ensure_degeneracy", False) or post_cfg.get("refine_degeneracy", False))
        if ensure_deg:
            loci = self._refine_locus_degeneracy(loci, verbose=verbose)

        target_bands = None
        if hasattr(self, "records") and self.records:
            best_rec = min(self.records, key=lambda r: r.get("raw_cost", float("inf")))
            target_bands = best_rec.get("target_bands")
        if not target_bands:
            target_bands = self.target_cfg.get("mode_indices") or self.target_cfg.get("target_bands") or [4, 5, 6]

        pol = str(self.target_cfg.get("polarization", "te")).lower()
        post_cfg = getattr(self, "post_cfg", {}) or {}
        post_vg = post_cfg.get("group_velocity", {}) if isinstance(post_cfg.get("group_velocity"), dict) else {}
        post_bd = post_cfg.get("band_diagram", {}) if isinstance(post_cfg.get("band_diagram"), dict) else (post_cfg.get("band_structure", {}) if isinstance(post_cfg.get("band_structure"), dict) else {})

        compute_vg = bool(post_vg.get("enabled", post_cfg.get("compute_group_velocity", False)))
        compute_bd = bool(post_bd.get("enabled", post_cfg.get("compute_band_diagram", True)))
        delta_k = float(post_vg.get("delta_k", post_cfg.get("delta_k", self.target_cfg.get("delta_k", 0.001))))

        worker_cand = (
            self.sim_cfg.get("parallel_workers")
            or self.opt_cfg.get("batch_size")
            or self.sim_cfg.get("cores")
            or 4
        )
        max_workers = min(max(1, int(worker_cand)), 24)

        for locus in loci:
            l_id = locus["locus_id"]
            locus_dir = self.output_dir / f"locus_{l_id:02d}"
            locus_dir.mkdir(parents=True, exist_ok=True)

            r1_pts = locus["r1"]
            r2_pts = locus["r2"]
            fom_pts = locus.get("fom", [])
            gap_pts = locus.get("residual_gap", [])
            n_pts = len(r1_pts)

            if not compute_bd and not compute_vg:
                if verbose:
                    print(f"Skipping per-point simulations for Locus #{l_id} (both band_diagram and group_velocity disabled in postprocessing).")
            else:
                if verbose:
                    sim_desc = "full band structure + group velocity" if (compute_bd and compute_vg) else ("band structure" if compute_bd else "group velocity")
                    print(f"Running {sim_desc} MPB simulations for Locus #{l_id} ({n_pts} points into '{locus_dir.name}/', {max_workers} concurrent workers)...")

                point_tasks = []
                for i in range(n_pts):
                    pt_idx = i + 1
                    pt_dir = locus_dir / f"pt_{pt_idx:02d}"
                    pt_dir.mkdir(parents=True, exist_ok=True)

                    p_entry = {}
                    if len(self.param_names) >= 1:
                        p_entry[self.param_names[0]] = float(r1_pts[i])
                    if len(self.param_names) >= 2:
                        p_entry[self.param_names[1]] = float(r2_pts[i])

                    t_norm = float(i) / max(n_pts - 1, 1)
                    fom_val = float(fom_pts[i]) if i < len(fom_pts) else 0.0
                    gap_val = float(gap_pts[i]) if i < len(gap_pts) else 0.0
                    point_tasks.append((pt_idx, t_norm, p_entry, fom_val, gap_val, pt_dir))

                def run_single_point(task_args):
                    pt_idx, t_norm, p_dict, fom_val, gap_val, pt_dir = task_args
                    bs_dir = pt_dir / "band_structure"
                    vg_dir = pt_dir / "group_velocity"

                    # 1. Full band structure simulation (k-path)
                    if compute_bd:
                        bs_dir.mkdir(parents=True, exist_ok=True)
                        bs_params = {**self.fixed_params, **p_dict}
                        bs_params["display_symmetry?"] = "true"
                        bs_params["display_group_velocity?"] = "false"
                        bs_params["only_gamma?"] = "false"
                        bs_params["delta_k_mode?"] = "false"
                        bs_params[f"run-{pol}?"] = "true"

                        run_hpc(
                            script=self._get_script_path(),
                            mpb_command_line_params=bs_params,
                            use_mpi=False,
                            cores=self.sim_cfg.get("cores", 1),
                            wd=bs_dir,
                            auto_extract=True,
                            auto_plot=False,
                            only_gamma=False,
                            verbose=False,
                        )

                        # Move files from bs_dir/output to bs_dir if output subdirectory exists
                        raw_out_dir = bs_dir / "output"
                        if raw_out_dir.is_dir():
                            for item in raw_out_dir.iterdir():
                                dest = bs_dir / item.name
                                if dest.exists():
                                    if dest.is_dir():
                                        shutil.rmtree(dest)
                                    else:
                                        dest.unlink()
                                shutil.move(str(item), str(dest))
                            shutil.rmtree(raw_out_dir, ignore_errors=True)

                        # Generate clean band structure plot and epsilon map
                        try:
                            from .plotter import plot_band_structure, plot_epsilon
                            plot_band_structure(
                                data_path=bs_dir,
                                output_path=bs_dir / "band_structure.png",
                                bands=target_bands,
                                polarization=pol,
                                highlight_gaps=False,
                                style="light",
                                verbose=False,
                            )
                            for h5_candidate in list(bs_dir.glob("*-epsilon.h5")):
                                if not h5_candidate.name.endswith(".converted.h5"):
                                    plot_epsilon(
                                        h5_path=h5_candidate,
                                        output_path=bs_dir / "epsilon_map.png",
                                        rectify=True,
                                        verbose=False,
                                        plane="both",
                                    )
                                    break
                        except Exception:
                            pass

                    # 2. Single-point group velocity simulation at k = (delta_k, 0, 0)
                    vg_val = 0.0
                    if compute_vg:
                        vg_dir.mkdir(parents=True, exist_ok=True)
                        vg_params = {**self.fixed_params, **p_dict}
                        vg_params["display_symmetry?"] = "false"
                        vg_params["display_group_velocity?"] = "true"
                        vg_params["only_gamma?"] = "false"
                        vg_params["delta_k_mode?"] = "true"
                        vg_params["delta_k"] = delta_k
                        vg_params["delta-k"] = delta_k
                        vg_params[f"run-{pol}?"] = "true"

                        run_hpc(
                            script=self._get_script_path(),
                            mpb_command_line_params=vg_params,
                            use_mpi=False,
                            cores=self.sim_cfg.get("cores", 1),
                            wd=vg_dir,
                            auto_extract=True,
                            auto_plot=False,
                            only_gamma=False,
                            verbose=False,
                        )

                        # Move files from vg_dir/output to vg_dir
                        raw_vg_out = vg_dir / "output"
                        if raw_vg_out.is_dir():
                            for item in raw_vg_out.iterdir():
                                dest = vg_dir / item.name
                                if dest.exists():
                                    if dest.is_dir():
                                        shutil.rmtree(dest)
                                    else:
                                        dest.unlink()
                                shutil.move(str(item), str(dest))
                            shutil.rmtree(raw_vg_out, ignore_errors=True)

                        # Extract group velocity at delta_k
                        vg_log = vg_dir / "output.out"
                        if vg_log.is_file():
                            from .extractor import extract_group_velocities
                            vg_data = extract_group_velocities(output_path=vg_log, save_data=True, verbose=False)
                            flat_recs = vg_data.get("flat_records", [])
                            target_vgs = []
                            for rec_v in flat_recs:
                                p_lower = rec_v.get("parity", "").lower()
                                is_match = (
                                    p_lower == pol
                                    or (pol == "zeven" and p_lower == "te")
                                    or (pol == "te" and p_lower == "zeven")
                                    or (pol == "zodd" and p_lower == "tm")
                                    or (pol == "tm" and p_lower == "zodd")
                                )
                                if is_match and int(rec_v.get("band", 0)) in target_bands:
                                    vx_val = float(rec_v.get("vx", 0.0))
                                    if np.isnan(vx_val):
                                        vx_val = float(rec_v.get("vg_mag", 0.0))
                                    target_vgs.append(vx_val)
                            if target_vgs:
                                vg_val = float(max(target_vgs))

                    # Save point_info.json in pt_dir
                    pt_info = {
                        "locus_id": l_id,
                        "point_index": pt_idx,
                        "t_normalized": t_norm,
                        "params": p_dict,
                        "group_velocity": vg_val if compute_vg else None,
                        "delta_k": delta_k if compute_vg else None,
                        "predicted_fom": fom_val,
                        "residual_gap": gap_val,
                        "target_bands": target_bands,
                        "is_valid": bool(locus.get("is_valid", [True] * n_pts)[pt_idx - 1]) if "is_valid" in locus and (pt_idx - 1) < len(locus["is_valid"]) else True,
                        "validity_status": str(locus.get("validity_status", ["PASSED"] * n_pts)[pt_idx - 1]) if "validity_status" in locus and (pt_idx - 1) < len(locus["validity_status"]) else "PASSED",
                    }
                    if "r1_unrefined" in locus and (pt_idx - 1) < len(locus["r1_unrefined"]):
                        pt_info["unrefined_params"] = {
                            self.param_names[0]: locus["r1_unrefined"][pt_idx - 1],
                        }
                        if len(self.param_names) >= 2:
                            pt_info["unrefined_params"][self.param_names[1]] = locus["r2_unrefined"][pt_idx - 1]

                    with open(pt_dir / "point_info.json", "w") as f:
                        json.dump(pt_info, f, indent=2)

                    return pt_idx - 1, vg_val

                results = []
                pbar = tqdm(total=n_pts, desc=f"Simulating Locus #{l_id}", unit="pt", disable=not verbose)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(run_single_point, task): task[0] for task in point_tasks}
                    for f in as_completed(futures):
                        res = f.result()
                        results.append(res)
                        idx_0, vg_v = res
                        pbar.set_postfix({"pt": f"{idx_0+1:02d}/{n_pts:02d}", "vg": f"{vg_v:.4f}"})
                        pbar.update(1)
                pbar.close()

                if compute_vg:
                    vg_ordered = [0.0] * n_pts
                    for idx_0, vg_v in results:
                        vg_ordered[idx_0] = vg_v
                    locus["group_velocity"] = vg_ordered
                    locus["vg"] = vg_ordered

            # Save per-locus CSV and 3-panel profile figure
            from .locus import export_loci_to_csv, plot_locus_profiles
            locus_csv = locus_dir / "bo_locus.csv"
            export_loci_to_csv([locus], locus_csv)
            if verbose:
                print(f"Saved Locus #{l_id} table to '{locus_csv}'")

            locus_fig = locus_dir / "bo_locus_profile.png"
            plot_locus_profiles(
                [locus],
                param_names=self.param_names,
                param_bounds=self.param_bounds,
                output_path=locus_fig,
                title=f"Optimal Locus #{l_id} ({self.target_cfg.get('irreps_str', 'Target')})",
            )

        return loci

    def _get_script_path(self) -> Path:
        script_cfg = self.sim_cfg.get("ctl_script", "example.ctl")
        w_path = (self.work_dir / script_cfg).resolve()
        if w_path.is_file():
            return w_path
        path = Path(script_cfg)
        if path.is_file():
            return path.resolve()
        return Path(script_cfg)

    def _run_validation(self, best_params: Dict[str, float]) -> None:
        print("\nExecuting final validation simulation across FULL k-path with optimal parameters...")
        combined = {**self.fixed_params, **best_params}
        combined["display_symmetry?"] = "true"
        pol = self.target_cfg.get("polarization", "te").lower()
        combined[f"run-{pol}?"] = "true"

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

        # Determine target bands (e.g. [4, 5, 6]) to focus validation band structure plot
        target_bands = None
        if hasattr(self, "records") and self.records:
            best_rec = min(self.records, key=lambda r: r.get("raw_cost", float("inf")))
            target_bands = best_rec.get("target_bands")
        if not target_bands:
            target_bands = self.target_cfg.get("mode_indices") or self.target_cfg.get("target_bands")

        output_dir = Path(self.output_dir) / "output"
        pol = self.target_cfg.get("polarization", "te").lower()
        if target_bands and output_dir.is_dir():
            try:
                from .plotter import plot_band_structure
                plot_band_structure(
                    data_path=output_dir,
                    output_path=output_dir / "band_structure.png",
                    bands=target_bands,
                    polarization=pol,
                    highlight_gaps=False,
                    style="light"
                )
            except Exception as e:
                print(f"Note: Could not adjust validation band structure plot limits: {e}")

        if res.returncode == 0:
            print(f"Validation complete! Optimal figures saved inside '{self.output_dir}'")
        else:
            print("Validation run finished with non-zero exit code.")


    def plot_only(self) -> None:
        """Loads existing simulation data and re-generates all output plots without running MPB simulations."""
        if not self.json_file.is_file() and not self.data_file.is_file():
            print(f"Error: No existing simulation data found in '{self.output_dir}' to plot.")
            return

        print("==================================================================")
        print("Re-plotting figures from previous simulation data...")
        print(f"Working Directory:   '{self.work_dir}'")
        print(f"Output Directory:    '{self.output_dir}'")
        print("==================================================================")

        if self.json_file.is_file():
            try:
                with open(self.json_file, "r") as f:
                    self.records = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load JSON data from {self.json_file}: {e}")

        # Re-populate optimizer space with evaluated points
        xi_list = []
        yi_list = []
        mode = self.opt_cfg.get("objective_mode", "log")
        target_cost = self.target_cfg.get("target_cost")

        for r in self.records:
            if "params" in r and "raw_cost" in r:
                pt = [r["params"][name] for name in self.param_names if name in r["params"]]
                if len(pt) == len(self.param_names):
                    xi_list.append(pt)
                    raw_c = float(r["raw_cost"])
                    eff_c = target_cost if (target_cost is not None and raw_c < target_cost) else raw_c
                    if mode == "log":
                        y_val = float(np.log10(max(eff_c, 1e-12)))
                    else:
                        y_val = float(eff_c)
                    yi_list.append(y_val)

        if xi_list and yi_list:
            self.optimizer.tell(xi_list, yi_list)

        self._plot_convergence()
        self._plot_surrogate_map(final=False, verbose=True)
        print(f"Plotting complete! All figures saved inside '{self.output_dir}'")

    def postprocess_only(self) -> None:
        """Loads existing simulation data/model and executes only the postprocessing pipeline (locus extraction, refinement, and simulations)."""
        if not self.json_file.is_file() and not self.data_file.is_file() and not self.model_file.is_file():
            print(f"Error: No existing optimization data or model found in '{self.output_dir}' to postprocess.")
            return

        print("==================================================================")
        print("Running Postprocessing Pipeline on Existing Optimization Output...")
        print(f"Working Directory:   '{self.work_dir}'")
        print(f"Output Directory:    '{self.output_dir}'")
        print("==================================================================")

        # Load existing trained model if available
        if self.model_file.is_file():
            try:
                self.optimizer = joblib.load(self.model_file)
            except Exception as e:
                print(f"Notice: Could not load {self.model_file}, will reconstruct: {e}")

        if self.json_file.is_file():
            try:
                with open(self.json_file, "r") as f:
                    self.records = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load JSON data from {self.json_file}: {e}")

        if not hasattr(self.optimizer, "models") or not self.optimizer.models:
            xi_list = []
            yi_list = []
            mode = self.opt_cfg.get("objective_mode", "log")
            target_cost = self.target_cfg.get("target_cost")

            for r in self.records:
                if "params" in r and "raw_cost" in r:
                    pt = [r["params"][name] for name in self.param_names if name in r["params"]]
                    if len(pt) == len(self.param_names):
                        xi_list.append(pt)
                        raw_c = float(r["raw_cost"])
                        eff_c = target_cost if (target_cost is not None and raw_c < target_cost) else raw_c
                        if mode == "log":
                            y_val = float(np.log10(max(eff_c, 1e-12)))
                        else:
                            y_val = float(eff_c)
                        yi_list.append(y_val)

            if xi_list and yi_list:
                self.optimizer.tell(xi_list, yi_list)

        self._plot_convergence(verbose=True)
        self._plot_surrogate_map(final=True, verbose=True)
        print(f"Postprocessing complete! All outputs saved inside '{self.output_dir}'")


def run_bo(config_path: Union[str, os.PathLike], work_dir: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
    """
    Main programmatic API function to run Bayesian Optimization.
    """
    config = load_bo_config(config_path)
    if work_dir:
        config["simulation"]["work_dir"] = str(work_dir)

    opt = BayesianOptimizer(config)
    return opt.run()


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Run Photonic Crystal Bayesian Optimization & Postprocessing with Hydra.
    """
    raw_dict = OmegaConf.to_container(cfg, resolve=True)
    general_cfg = raw_dict.get("general", {})

    opt = BayesianOptimizer(raw_dict)

    if general_cfg.get("postprocess_only", False) or general_cfg.get("postprocess", False):
        opt.postprocess_only()
    elif general_cfg.get("plot_only", False) or general_cfg.get("plot", False):
        opt.plot_only()
    else:
        opt.run()


if __name__ == "__main__":
    main()
