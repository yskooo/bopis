"""Live dashboard mode: the bridge reports the newest run as it progresses."""

import json
import os
import tempfile
import unittest

from bopis import ui_server


def _run(root, name):
    path = os.path.join(root, name)
    os.makedirs(os.path.join(path, "calibration"))
    return path


class LiveRunTests(unittest.TestCase):
    def test_no_runs(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(ui_server.live_run(root), {"run": None})
            self.assertEqual(ui_server.live_run(os.path.join(root, "x")), {"run": None})

    def test_running_then_finished(self):
        with tempfile.TemporaryDirectory() as root:
            _run(root, "20260101T000000Z")  # older, ignored
            run = _run(root, "20260102T000000Z")
            with open(os.path.join(run, "progress.json"), "w") as handle:
                json.dump({"stage": "Stage 1: BOPIS search", "n_total": 30,
                           "backend": "sim"}, handle)
            with open(os.path.join(run, "calibration", "bo_log.csv"), "w") as handle:
                handle.write("iteration,source,config,energy_j,tokens_per_s,quality_f1,"
                             "gp_mu,gp_sigma,expected_improvement\n")
                handle.write("1,seed,t128_b1_Q4_K_M_g0_c4,50.5,20.0,0.8,,,\n")
                handle.write("2,bo,t256_b1_Q8_0_g0_c4,,,,40.1,3.2,1.5\n")

            live = ui_server.live_run(root)
            self.assertEqual(live["run"], "20260102T000000Z")
            self.assertFalse(live["finished"])
            self.assertIsNone(live["data_url"])
            self.assertEqual((live["n_total"], live["backend"]), (30, "sim"))
            self.assertEqual(len(live["evaluations"]), 2)
            first, second = live["evaluations"]
            self.assertEqual(first["config"], "t128_b1_Q4_K_M_g0_c4")
            self.assertIsNone(second["energy_j"])  # NaN -> null
            self.assertEqual(second["expected_improvement"], 1.5)
            json.dumps(live, allow_nan=False)

            open(os.path.join(run, "dashboard_data.js"), "w").close()
            live = ui_server.live_run(root)
            self.assertTrue(live["finished"])
            self.assertEqual(live["data_url"], "/api/live/data?run=20260102T000000Z")

    def test_find_llama_binary_prefers_vulkan(self):
        with tempfile.TemporaryDirectory() as root:
            for flavour in ("cpu", "vulkan"):
                os.makedirs(os.path.join(root, "tools", flavour))
                binary = os.path.join(root, "tools", flavour, "llama-server.exe")
                open(binary, "w").close()
            self.assertIn("vulkan", ui_server.find_llama_binary(root))

    def test_list_runs_and_prompts(self):
        with tempfile.TemporaryDirectory() as root:
            run = _run(root, "20260103T000000Z_ui-sim")
            os.makedirs(os.path.join(run, "dataset"))
            os.makedirs(os.path.join(root, "not-a-run"))  # no calibration dir
            with open(os.path.join(run, "calibration", "bo_log.csv"), "w") as handle:
                handle.write("iteration\n1\n2\n")
            with open(os.path.join(run, "dataset", "sample_3.csv"), "w") as handle:
                handle.write("prompt_index,prompt_id,task_type,in_proxy_subset,"
                             "estimated_prompt_tokens\n"
                             "1,dolly-1,open_qa,True,20\n"
                             "2,dolly-2,summarization,False,300\n"
                             "3,synthetic-9,closed_qa,True,40\n")
            runs = ui_server.list_runs(root)
            self.assertEqual([r["run"] for r in runs], ["20260103T000000Z_ui-sim"])
            self.assertEqual(runs[0]["n_evals"], 2)
            self.assertFalse(runs[0]["finished"])

            class Prompt:
                prompt_id, instruction = "dolly-1", "Why is the sky blue?"

            prompts = ui_server.run_prompts("20260103T000000Z_ui-sim", root, [Prompt()])
            self.assertEqual(prompts["n_sample"], 3)
            self.assertEqual([p["prompt_id"] for p in prompts["proxy"]],
                             ["dolly-1", "synthetic-9"])
            self.assertEqual(prompts["proxy"][0]["instruction"], "Why is the sky blue?")
            self.assertIsNone(prompts["proxy"][1]["instruction"])
            self.assertIsNone(ui_server.run_prompts("missing", root))

    def test_study_command(self):
        instruments = ui_server.Instruments.__new__(ui_server.Instruments)
        instruments.dolly, instruments.data_dir, instruments.source = [], "data", None
        instruments.llama_binary = None
        instruments.model_paths = {}
        sim = instruments.study_command({"preset": "sim", "iterations": 500,
                                         "proxy_size": 2})
        self.assertIn("--synthetic", sim)  # no Dolly pool loaded
        self.assertEqual(sim[sim.index("--iterations") + 1], "60")  # clamped
        self.assertEqual(sim[sim.index("--proxy-size") + 1], "8")  # one per task type
        with self.assertRaises(ValueError):
            instruments.study_command({"preset": "real"})  # no binary
        instruments.llama_binary = "llama-server.exe"
        instruments.model_paths = {"qwen2.5-1.5b:Q4_K_M": "m.gguf"}
        real = instruments.study_command({"preset": "real"})
        self.assertIn("--model-aware", real)
        self.assertEqual(real[real.index("--llama-port") + 1], "8084")  # not the chat's
        self.assertIn("qwen2.5-1.5b:Q4_K_M=m.gguf", real)


if __name__ == "__main__":
    unittest.main()
