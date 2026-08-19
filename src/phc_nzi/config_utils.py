"""
Centralized Configuration Utilities for Deep Merging & Priority Resolution.
Implements the 4-tier deterministic configuration hierarchy across Hydra modules and workflow steps.
"""

from typing import Any, Dict, List, Optional, Union
from omegaconf import OmegaConf, DictConfig


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively deep merges an override dictionary into a base dictionary.
    
    Rules:
    - If a key exists in both and both values are dictionaries, merge them recursively.
    - If a key exists in override and is not a dictionary (or base is not a dictionary),
      override's value replaces base's value (unless override's value is None).
    - Preserves all unmentioned sibling keys in base.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override if override is not None else base

    result = base.copy()
    for key, value in override.items():
        if value is None:
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_step_config(
    global_cfg: Union[DictConfig, Dict[str, Any]],
    step_name: str,
    step_specific_cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Constructs the active configuration dictionary for a workflow step following
    the 4-tier priority hierarchy:
      Tier 1 (CLI) > Tier 2 (Step Context Overrides) > Tier 3 (Modular Subsystems) > Tier 4 (Base Defaults)

    Parameters:
    -----------
    global_cfg : DictConfig or dict
        Composed Hydra master configuration (containing simulation, parameters, target, optimizer, postprocessing, workflow).
    step_name : str
        Name of the workflow step (e.g. "step1_2d_screening", "step2_3d_screening", "step3_optimization").
    step_specific_cfg : DictConfig or dict, optional
        Step-specific override dictionary from workflow configuration.

    Returns:
    --------
    dict:
        Resolved, fully deep-merged dictionary ready for direct use in simulation runners.
    """
    if isinstance(global_cfg, DictConfig):
        base_dict = OmegaConf.to_container(global_cfg, resolve=True)
    else:
        base_dict = dict(global_cfg)

    if step_specific_cfg is not None:
        if isinstance(step_specific_cfg, DictConfig):
            step_dict = OmegaConf.to_container(step_specific_cfg, resolve=True)
        else:
            step_dict = dict(step_specific_cfg)

        # 1. Deep merge recognized subsystem blocks defined directly inside step configuration
        # (e.g. step3_optimization.postprocessing, step3_optimization.optimizer, step3_optimization.parameters)
        for subsystem in ["postprocessing", "optimizer", "parameters", "simulation", "target"]:
            if subsystem in step_dict and isinstance(step_dict[subsystem], dict):
                base_dict[subsystem] = deep_merge(base_dict.get(subsystem, {}), step_dict[subsystem])

        # 2. Map canonical parameter shortcuts from step configuration into their respective subsystems
        # Fixed parameters (sz, num_bands, h, resolution, res_z)
        fixed_params = base_dict.setdefault("parameters", {}).setdefault("fixed", {})
        if "sz" in step_dict and step_dict["sz"] is not None:
            sz_val = step_dict["sz"]
            fixed_params["sz"] = 4.0 if str(sz_val).lower() == "no-size" else float(sz_val)
        if "num_bands" in step_dict and step_dict["num_bands"] is not None:
            fixed_params["num-bands"] = int(step_dict["num_bands"])
            fixed_params["num_bands"] = int(step_dict["num_bands"])
        if "resolution" in step_dict and step_dict["resolution"] is not None:
            fixed_params["resolution"] = int(step_dict["resolution"])
        if "res_z" in step_dict and step_dict["res_z"] is not None:
            fixed_params["res-z"] = int(step_dict["res_z"])
            fixed_params["res_z"] = int(step_dict["res_z"])

        # Optimizer shortcuts
        opt_cfg = base_dict.setdefault("optimizer", {})
        if "initial_points" in step_dict and step_dict["initial_points"] is not None:
            opt_cfg.setdefault("initial_sampling", {})["initial_points"] = int(step_dict["initial_points"])
            opt_cfg.setdefault("initial_sampling", {})["n_initial_points"] = int(step_dict["initial_points"])
        if "max_iterations" in step_dict and step_dict["max_iterations"] is not None:
            opt_cfg.setdefault("iterations", {})["max_iterations"] = int(step_dict["max_iterations"])
        if "batch_size" in step_dict and step_dict["batch_size"] is not None:
            opt_cfg.setdefault("iterations", {})["batch_size"] = int(step_dict["batch_size"])
            opt_cfg.setdefault("initial_sampling", {})["batch_size"] = int(step_dict["batch_size"])

        # Postprocessing shortcuts
        post_cfg = base_dict.setdefault("postprocessing", {})
        if "refinement_tolerance" in step_dict and step_dict["refinement_tolerance"] is not None:
            post_cfg.setdefault("refinement", {})["tolerance"] = float(step_dict["refinement_tolerance"])
        if "sample_points" in step_dict and step_dict["sample_points"] is not None:
            post_cfg.setdefault("locus", {})["sample_points"] = int(step_dict["sample_points"])
        if "locus_mode" in step_dict and step_dict["locus_mode"] is not None:
            post_cfg.setdefault("locus", {})["mode"] = str(step_dict["locus_mode"]).lower()

    return base_dict
