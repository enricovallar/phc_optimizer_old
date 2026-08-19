"""
Discovery Workflow Pipeline Orchestrator
Coordinates Steps 1 through 5 in an end-to-end autonomous discovery pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import hydra
from omegaconf import DictConfig, OmegaConf

from .step1_2d_screening import run_step1_2d_screening
from .step2_3d_screening import run_step2_3d_screening
from .step3_optimization import run_step3_optimization
from .step4_design_curves import run_step4_design_curves
from .step5_validation import run_step5_validation


class DiscoveryWorkflow:
    """
    Modular 5-Step Photonic Crystal Near-Zero-Index Discovery Engine.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.general_cfg = cfg.get("general", {})
        self.sim_cfg = cfg.get("simulation", {})
        self.params_cfg = cfg.get("parameters", {})
        self.target_cfg = cfg.get("target", {})
        self.wf_cfg = cfg.get("workflow", {})

        self.work_dir = Path(self.sim_cfg.get("work_dir", ".")).resolve()
        self.output_dir = Path(self.wf_cfg.get("output_dir", "workflow_output")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Robustly extract search parameters (supports dict 'search: {r1: [...], r2: [...]}' or list 'sweep: [...]')
        search_dict = self.params_cfg.get("search", self.params_cfg.get("sweep", {}))
        if isinstance(search_dict, (dict, DictConfig)):
            self.param_names = list(search_dict.keys())
            self.param_bounds = [list(b) for b in search_dict.values()]
        elif isinstance(search_dict, list):
            self.param_names = [p.name if hasattr(p, "name") else p["name"] for p in search_dict]
            self.param_bounds = [p.bounds if hasattr(p, "bounds") else p["bounds"] for p in search_dict]
        else:
            self.param_names = ["r1", "r2"]
            self.param_bounds = [[0.2, 0.4], [0.2, 0.4]]

        self.fixed_params = dict(self.params_cfg.get("fixed", {}))

    def run(self) -> Dict[str, Any]:
        """Runs the entire workflow pipeline or a specific step specified in config."""
        step_to_run = str(self.wf_cfg.get("step", "all")).lower()
        interactive = bool(self.wf_cfg.get("interactive", True))

        results = {}

        if step_to_run in ["1", "all"]:
            res1 = run_step1_2d_screening(
                cfg=self.cfg,
                work_dir=self.work_dir,
                output_dir=self.output_dir,
                param_names=self.param_names,
                param_bounds=self.param_bounds,
                fixed_params=self.fixed_params,
                sim_cfg=self.sim_cfg,
            )
            results["step1"] = res1
            if step_to_run == "1":
                return results

            if interactive:
                print(f"\n[Step 1 Complete] Top 2D candidate: Bands {res1['best_triplet']['bands']} ({res1['best_triplet']['dominant_irreps']})")

        if step_to_run in ["2", "all"]:
            t_2d = results.get("step1", {}).get("best_triplet")
            res2 = run_step2_3d_screening(
                cfg=self.cfg,
                work_dir=self.work_dir,
                output_dir=self.output_dir,
                param_names=self.param_names,
                param_bounds=self.param_bounds,
                fixed_params=self.fixed_params,
                sim_cfg=self.sim_cfg,
                target_2d_triplet=t_2d,
            )
            results["step2"] = res2
            if step_to_run == "2":
                return results

        if step_to_run in ["3", "3_postprocess", "3_postprocessing", "postprocess", "postprocessing", "all"]:
            t_3d = results.get("step2", {}).get("best_triplet", {})
            t_irreps = t_3d.get("target_irreps", ["A_2", "E", "E"])
            t_occs = t_3d.get("irrep_occurrences", [2, 3, 3])
            t_bands = t_3d.get("bands", [8, 9, 10])

            res3 = run_step3_optimization(
                cfg=self.cfg,
                work_dir=self.work_dir,
                output_dir=self.output_dir,
                param_names=self.param_names,
                param_bounds=self.param_bounds,
                fixed_params=self.fixed_params,
                sim_cfg=self.sim_cfg,
                target_irreps=t_irreps,
                irrep_occurrences=t_occs,
                target_modes=t_bands,
            )
            results["step3"] = res3
            if step_to_run in ["3", "3_postprocess", "3_postprocessing", "postprocess", "postprocessing"]:
                return results

        if step_to_run in ["4", "all"]:
            sw_rec = results.get("step3", {}).get("sweep_records")
            res4 = run_step4_design_curves(
                cfg=self.cfg,
                work_dir=self.work_dir,
                output_dir=self.output_dir,
                param_names=self.param_names,
                sweep_records=sw_rec,
            )
            results["step4"] = res4
            if step_to_run == "4":
                return results

        if step_to_run in ["5", "all"]:
            dc_res = results.get("step4")
            res5 = run_step5_validation(
                cfg=self.cfg,
                work_dir=self.work_dir,
                output_dir=self.output_dir,
                param_names=self.param_names,
                design_curves_result=dc_res,
            )
            results["step5"] = res5

        return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI Entrypoint for Photonic Crystal Discovery Workflow."""
    workflow = DiscoveryWorkflow(cfg)
    workflow.run()


if __name__ == "__main__":
    main()
