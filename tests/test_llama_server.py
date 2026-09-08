"""The llama-server adapter: launch flags, response parsing, log parsing.

No server and no model weights are required. The parts that need them
(``start``/``generate``) are exercised only for their failure paths; everything
that shapes a measurement — which flags go on the command line, how token counts
and timings are read out — is pure and tested directly.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from bopis import config_space as cs
from bopis.backends import BackendError
from bopis.backends.llama_server import LlamaServerBackend, parse_server_log
from bopis.config_space import ALL_LAYERS, Config


def backend(**kwargs) -> LlamaServerBackend:
    defaults = dict(
        binary="llama-server",
        model_paths={"Q4_K_M": "/models/m.Q4_K_M.gguf", "F16": "/models/m.F16.gguf"},
    )
    defaults.update(kwargs)
    return LlamaServerBackend(**defaults)


class TestLaunchArgs(unittest.TestCase):
    """Launch-time versus request-time parameters (amendment A-5)."""

    def setUp(self) -> None:
        # Point at real files so the existence check passes.
        self.tmp = tempfile.mkdtemp(prefix="bopis-gguf-")
        self.paths = {}
        for variant in ("F16", "Q4_K_M"):
            path = os.path.join(self.tmp, f"m.{variant}.gguf")
            with open(path, "wb") as handle:
                handle.write(b"\x00")
            self.paths[variant] = path
        self.backend = backend(model_paths=self.paths)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_precision_selects_the_gguf_file(self) -> None:
        args = self.backend.launch_args(Config(256, 1, "Q4_K_M", 14, 4))
        self.assertIn(self.paths["Q4_K_M"], args)
        self.assertNotIn(self.paths["F16"], args)

    def test_gpu_layers_are_a_launch_flag(self) -> None:
        args = self.backend.launch_args(Config(256, 1, "F16", 14, 4))
        self.assertIn("--n-gpu-layers", args)
        self.assertEqual(args[args.index("--n-gpu-layers") + 1], "14")

    def test_all_layers_resolves_against_model_depth(self) -> None:
        args = self.backend.launch_args(
            Config(256, 1, "F16", ALL_LAYERS, 4), total_layers=32
        )
        self.assertEqual(args[args.index("--n-gpu-layers") + 1], "32")

    def test_cpu_threads_are_a_launch_flag(self) -> None:
        args = self.backend.launch_args(Config(256, 1, "F16", 0, 8))
        self.assertEqual(args[args.index("--threads") + 1], "8")

    def test_batch_size_becomes_parallel_slots(self) -> None:
        args = self.backend.launch_args(Config(256, 4, "F16", 0, 4))
        self.assertEqual(args[args.index("--parallel") + 1], "4")

    def test_context_size_is_fixed_and_not_t(self) -> None:
        """Amendment A-1: t is n_predict, so it must NOT appear as --ctx-size."""
        config = Config(128, 1, "F16", 0, 4)
        args = self.backend.launch_args(config)
        self.assertEqual(
            args[args.index("--ctx-size") + 1], str(cs.FIXED_CTX_SIZE)
        )
        self.assertNotEqual(args[args.index("--ctx-size") + 1], str(config.t))

    def test_generation_cap_is_not_a_launch_flag(self) -> None:
        """t is per-request; it must not be baked into the server launch."""
        args = self.backend.launch_args(Config(1024, 1, "F16", 0, 4))
        self.assertNotIn("--n-predict", args)
        self.assertNotIn("--predict", args)

    def test_seed_is_fixed(self) -> None:
        args = backend(model_paths=self.paths, seed=7).launch_args(
            Config(256, 1, "F16", 0, 4)
        )
        self.assertEqual(args[args.index("--seed") + 1], "7")

    def test_extra_args_are_appended(self) -> None:
        args = backend(
            model_paths=self.paths, extra_args=["--flash-attn", "--mlock"]
        ).launch_args(Config(256, 1, "F16", 0, 4))
        self.assertIn("--flash-attn", args)
        self.assertIn("--mlock", args)

    def test_missing_variant_fails_loudly(self) -> None:
        """A variant with no GGUF must raise, never silently substitute."""
        with self.assertRaises(BackendError) as caught:
            self.backend.launch_args(Config(256, 1, "F32", 0, 4))
        self.assertIn("F32", str(caught.exception))

    def test_missing_file_fails_loudly(self) -> None:
        broken = backend(model_paths={"F16": "/nonexistent/model.gguf"})
        with self.assertRaises(BackendError) as caught:
            broken.launch_args(Config(256, 1, "F16", 0, 4))
        self.assertIn("not found", str(caught.exception))

    def test_two_configs_differing_only_in_t_share_a_launch(self) -> None:
        """Which is why t does not require a server restart."""
        first = self.backend.launch_args(Config(128, 1, "F16", 14, 4))
        second = self.backend.launch_args(Config(1024, 1, "F16", 14, 4))
        self.assertEqual(first, second)

    def test_two_configs_differing_in_precision_do_not(self) -> None:
        first = self.backend.launch_args(Config(256, 1, "F16", 14, 4))
        second = self.backend.launch_args(Config(256, 1, "Q4_K_M", 14, 4))
        self.assertNotEqual(first, second)


class TestCompletionParsing(unittest.TestCase):
    """Parsing captured ``/completion`` payloads."""

    def parse(self, payload, config=Config(256, 1, "F16", 14, 4), wall=1.5):
        return LlamaServerBackend._parse_completion(payload, wall, config)

    def test_reads_token_counts_and_timings(self) -> None:
        result = self.parse(
            {
                "content": "Paris is the capital of France.",
                "tokens_evaluated": 12,
                "tokens_predicted": 8,
                "stopped_eos": True,
                "timings": {
                    "prompt_n": 12,
                    "prompt_ms": 45.5,
                    "predicted_n": 8,
                    "predicted_ms": 320.25,
                },
            }
        )
        self.assertEqual(result.n_prompt_tokens, 12)
        self.assertEqual(result.n_generated_tokens, 8)
        self.assertAlmostEqual(result.prompt_ms, 45.5, places=6)
        self.assertAlmostEqual(result.predicted_ms, 320.25, places=6)
        self.assertIn("Paris", result.text)
        self.assertFalse(result.truncated)

    def test_prefill_decode_split_enables_separate_energy(self) -> None:
        """Amendment A-35 depends on this split being available."""
        result = self.parse(
            {
                "content": "x",
                "tokens_predicted": 4,
                "timings": {"prompt_ms": 100.0, "predicted_ms": 300.0},
            }
        )
        total = result.prompt_ms + result.predicted_ms
        self.assertAlmostEqual(result.prompt_ms / total, 0.25, places=9)

    def test_decode_throughput_excludes_prefill(self) -> None:
        result = self.parse(
            {
                "content": "x",
                "tokens_predicted": 10,
                "timings": {"prompt_ms": 500.0, "predicted_ms": 1000.0},
            }
        )
        self.assertAlmostEqual(result.decode_tokens_per_s, 10.0, places=9)

    def test_falls_back_to_timings_for_token_counts(self) -> None:
        result = self.parse(
            {"content": "x", "timings": {"prompt_n": 5, "predicted_n": 3}}
        )
        self.assertEqual(result.n_prompt_tokens, 5)
        self.assertEqual(result.n_generated_tokens, 3)

    def test_stopped_limit_marks_truncation(self) -> None:
        """A response cut off at n_predict must be visible in the logs."""
        result = self.parse(
            {"content": "x", "tokens_predicted": 5, "stopped_limit": True}
        )
        self.assertTrue(result.truncated)

    def test_reaching_the_cap_marks_truncation(self) -> None:
        config = Config(128, 1, "F16", 14, 4)
        result = self.parse({"content": "x", "tokens_predicted": 128}, config=config)
        self.assertTrue(result.truncated)

    def test_stopping_short_of_the_cap_is_not_truncation(self) -> None:
        config = Config(1024, 1, "F16", 14, 4)
        result = self.parse({"content": "x", "tokens_predicted": 40}, config=config)
        self.assertFalse(result.truncated)

    def test_missing_timings_are_none_not_zero(self) -> None:
        """None means 'not reported'; zero would be a measurement."""
        result = self.parse({"content": "x", "tokens_predicted": 4})
        self.assertIsNone(result.prompt_ms)
        self.assertIsNone(result.predicted_ms)
        self.assertIsNone(result.decode_tokens_per_s)

    def test_empty_payload_does_not_crash(self) -> None:
        result = self.parse({})
        self.assertEqual(result.n_generated_tokens, 0)
        self.assertEqual(result.text, "")
        self.assertEqual(result.tokens_per_s, 0.0)

    def test_malformed_timings_are_tolerated(self) -> None:
        result = self.parse({"content": "x", "timings": "not a dict"})
        self.assertIsNone(result.prompt_ms)

    def test_tokens_per_second_from_wall_clock(self) -> None:
        result = self.parse({"content": "x", "tokens_predicted": 30}, wall=2.0)
        self.assertAlmostEqual(result.tokens_per_s, 15.0, places=9)


class TestLogParsing(unittest.TestCase):
    """Fallback parsing of llama.cpp's stderr timing block."""

    SAMPLE = """
llama_model_loader: loaded meta data with 26 key-value pairs
llm_load_tensors: offloaded 32/33 layers to GPU

llama_print_timings:        load time =    1523.44 ms
llama_print_timings:      sample time =      12.31 ms /   128 runs
llama_print_timings: prompt eval time =     245.67 ms /    42 tokens
llama_print_timings:        eval time =    4321.09 ms /   127 runs
llama_print_timings:       total time =    6102.51 ms
"""

    def test_extracts_timings(self) -> None:
        found = parse_server_log(self.SAMPLE)
        self.assertAlmostEqual(found["load_time_ms"], 1523.44, places=4)
        self.assertAlmostEqual(found["prompt_ms"], 245.67, places=4)
        self.assertAlmostEqual(found["predicted_ms"], 4321.09, places=4)
        self.assertAlmostEqual(found["total_ms"], 6102.51, places=4)

    def test_extracts_token_counts(self) -> None:
        found = parse_server_log(self.SAMPLE)
        self.assertEqual(found["n_prompt_tokens"], 42.0)
        self.assertEqual(found["n_generated_tokens"], 127.0)

    def test_distinguishes_eval_from_prompt_eval(self) -> None:
        """The two lines both contain 'eval time'; they must not be confused."""
        found = parse_server_log(self.SAMPLE)
        self.assertNotAlmostEqual(found["prompt_ms"], found["predicted_ms"])

    def test_empty_input_yields_nothing(self) -> None:
        self.assertEqual(parse_server_log(""), {})

    def test_unrelated_text_yields_nothing(self) -> None:
        self.assertEqual(parse_server_log("hello world\nno timings here"), {})


class TestLifecycleGuards(unittest.TestCase):
    def test_generate_before_start_raises(self) -> None:
        with self.assertRaises(BackendError):
            backend().generate("hi", Config(256, 1, "F16", 0, 4))

    def test_count_tokens_without_a_server_returns_none(self) -> None:
        self.assertIsNone(backend().count_tokens("hello"))

    def test_missing_binary_reports_actionably(self) -> None:
        tmp = tempfile.mkdtemp(prefix="bopis-gguf-")
        try:
            path = os.path.join(tmp, "m.F16.gguf")
            with open(path, "wb") as handle:
                handle.write(b"\x00")
            broken = backend(
                binary="definitely-not-a-real-binary-xyz",
                model_paths={"F16": path},
            )
            with self.assertRaises(BackendError) as caught:
                broken.start(Config(256, 1, "F16", 0, 4))
            message = str(caught.exception)
            self.assertIn("could not launch", message)
            # Must point the reader at a way forward.
            self.assertIn("--backend sim", message)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_stop_without_start_is_safe(self) -> None:
        backend().stop()  # must not raise

    def test_describe_records_determinism_caveat(self) -> None:
        """Amendment A-31: determinism does not hold across configurations."""
        payload = backend().describe()
        self.assertEqual(payload["backend"], "llama-server")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_k"], 1)
        self.assertIn("determinism_note", payload)
        self.assertIn("across configurations", payload["determinism_note"])

    def test_base_url(self) -> None:
        self.assertEqual(
            backend(host="10.0.0.5", port=9999).base_url, "http://10.0.0.5:9999"
        )


if __name__ == "__main__":
    unittest.main()
