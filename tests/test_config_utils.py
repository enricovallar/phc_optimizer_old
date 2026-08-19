import unittest
from omegaconf import OmegaConf
from phc_nzi.config_utils import deep_merge, resolve_step_config


class TestConfigUtils(unittest.TestCase):
    def test_deep_merge_basic(self):
        base = {
            "postprocessing": {
                "locus": {"mode": "polar", "sample_points": 40},
                "refinement": {"enabled": True, "tolerance": 1e-5}
            },
            "optimizer": {"max_iterations": 5}
        }
        override = {
            "postprocessing": {
                "locus": {"mode": "cartesian"}
            }
        }
        merged = deep_merge(base, override)
        
        # Mode is updated
        self.assertEqual(merged["postprocessing"]["locus"]["mode"], "cartesian")
        # Sibling key in locus is preserved
        self.assertEqual(merged["postprocessing"]["locus"]["sample_points"], 40)
        # Sibling block in postprocessing is preserved
        self.assertTrue(merged["postprocessing"]["refinement"]["enabled"])
        self.assertEqual(merged["postprocessing"]["refinement"]["tolerance"], 1e-5)
        # Sibling block in root is preserved
        self.assertEqual(merged["optimizer"]["max_iterations"], 5)

    def test_deep_merge_none_handling(self):
        base = {"a": 1, "b": {"c": 2}}
        override = {"a": None, "b": {"c": 3, "d": None}}
        merged = deep_merge(base, override)
        self.assertEqual(merged["a"], 1)
        self.assertEqual(merged["b"]["c"], 3)
        self.assertNotIn("d", merged["b"])

    def test_resolve_step_config_hierarchy(self):
        global_cfg = OmegaConf.create({
            "parameters": {"fixed": {"num-bands": 10, "sz": "no-size"}},
            "optimizer": {"initial_sampling": {"initial_points": 100}, "iterations": {"max_iterations": 10}},
            "postprocessing": {
                "locus": {"mode": "auto", "sample_points": 30},
                "refinement": {"tolerance": 1e-4}
            },
            "workflow": {
                "step3_optimization": {
                    "sz": 4,
                    "num_bands": 14,
                    "sample_points": 50,
                    "locus_mode": "polar",
                    "postprocessing": {
                        "refinement": {"tolerance": 1e-6}
                    }
                }
            }
        })

        s3_cfg = global_cfg.workflow.step3_optimization
        resolved = resolve_step_config(global_cfg, "step3_optimization", s3_cfg)

        # Step-specific shortcuts correctly mapped
        self.assertEqual(resolved["parameters"]["fixed"]["sz"], 4.0)
        self.assertEqual(resolved["parameters"]["fixed"]["num-bands"], 14)
        self.assertEqual(resolved["postprocessing"]["locus"]["sample_points"], 50)
        self.assertEqual(resolved["postprocessing"]["locus"]["mode"], "polar")
        # Step-specific nested override deep-merged
        self.assertEqual(resolved["postprocessing"]["refinement"]["tolerance"], 1e-6)
        # Global untouched keys preserved
        self.assertEqual(resolved["optimizer"]["initial_sampling"]["initial_points"], 100)
        self.assertEqual(resolved["optimizer"]["iterations"]["max_iterations"], 10)


if __name__ == "__main__":
    unittest.main()
