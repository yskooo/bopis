"""Model discovery on disk, configuration-key parsing, and the chat's
replay/comparison endpoints.

No model is loaded: GGUF files are empty stand-ins with the official names, and
the comparison is exercised on the paths that need no llama-server (malformed
keys, missing files, missing binary). The live comparison was verified by hand
against real Qwen2.5 GGUFs; it needs minutes and gigabytes, so it is not a unit
test.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from bopis import config_space as cs
from bopis import gguf


def touch(directory: str, name: str, size: int = 10) -> str:
    path = os.path.join(directory, name)
    with open(path, "wb") as handle:
        handle.write(b"\0" * size)
    return path


class TestDiscovery(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_maps_official_names_to_keys(self) -> None:
        touch(self.dir, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        touch(self.dir, "qwen2.5-1.5b-instruct-fp16.gguf")
        touch(self.dir, "qwen2.5-3b-instruct-q8_0.gguf")
        touch(self.dir, "unrelated-model.gguf")
        found = gguf.discover(self.dir)
        self.assertEqual(
            set(found),
            {"qwen2.5-0.5b:Q4_K_M", "qwen2.5-1.5b:F16", "qwen2.5-3b:Q8_0"},
        )

    def test_split_files_are_summed_and_keyed_by_first_part(self) -> None:
        first = touch(self.dir, "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", 300)
        touch(self.dir, "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf", 200)
        self.assertEqual(gguf.discover(self.dir), {"qwen2.5-7b:Q4_K_M": first})
        self.assertEqual(gguf.total_size(first), 500)

    def test_incomplete_split_is_skipped(self) -> None:
        touch(self.dir, "qwen2.5-3b-instruct-fp16-00001-of-00002.gguf")
        self.assertEqual(gguf.discover(self.dir), {})
        with self.assertRaises(FileNotFoundError):
            gguf.total_size(
                os.path.join(self.dir, "qwen2.5-3b-instruct-fp16-00001-of-00002.gguf")
            )

    def test_unknown_rung_is_ignored(self) -> None:
        touch(self.dir, "qwen2.5-14b-instruct-q4_k_m.gguf")
        self.assertEqual(gguf.discover(self.dir), {})

    def test_missing_directory_is_empty(self) -> None:
        self.assertEqual(gguf.discover(os.path.join(self.dir, "nope")), {})


class TestParseKey(unittest.TestCase):
    def test_round_trips_every_configuration(self) -> None:
        for cfg in cs.full_space():
            self.assertEqual(cs.parse_key(cfg.key()), cfg)

    def test_pre_a40_keys_mean_the_default_model(self) -> None:
        cfg = cs.parse_key("t128_b1_Q4_K_M_g0_c4")
        self.assertEqual(cfg.m, cs.DEFAULT_M)
        self.assertEqual((cfg.t, cfg.p, cfg.g), (128, "Q4_K_M", 0))

    def test_rejects_malformed(self) -> None:
        self.assertIsNone(cs.parse_key("qwen2.5-3b_t128_Q4_K_M"))
        self.assertIsNone(cs.parse_key(""))


class TestReplayAndCompare(unittest.TestCase):
    """The chat server's Dolly replay and comparison, without a model."""

    @classmethod
    def setUpClass(cls) -> None:
        from bopis import ui_server

        cls.tmp = tempfile.TemporaryDirectory()
        rows = [
            {"instruction": "What is a merchant bank?", "context": "",
             "response": "A bank that deals in trade finance.", "category": "open_qa"},
            {"instruction": "Summarize this.", "context": "Some passage.",
             "response": "A summary.", "category": "summarization"},
        ]
        with open(os.path.join(cls.tmp.name, "databricks-dolly-15k.jsonl"), "w",
                  encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        cls.models = tempfile.TemporaryDirectory()
        touch(cls.models.name, "qwen2.5-0.5b-instruct-q4_k_m.gguf")

        def make(**kwargs):
            return ui_server.Instruments(
                llama_url="http://127.0.0.1:9",
                hwmon_url="http://127.0.0.1:9/data.json",
                idle_seconds=0,
                data_dir=cls.tmp.name,
                log=lambda _m: None,
                model_paths=gguf.discover(cls.models.name),
                **kwargs,
            )

        cls.make = staticmethod(make)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()
        cls.models.cleanup()

    def test_same_prompt_can_be_loaded_again_by_id(self) -> None:
        instruments = self.make()
        first = instruments.dolly_prompt()
        again = instruments.dolly_prompt(prompt_id=first["prompt_id"])
        self.assertEqual(first["prompt_id"], again["prompt_id"])
        self.assertEqual(first["reference"], again["reference"])
        self.assertIsNone(instruments.dolly_prompt(prompt_id="dolly-99999"))

    def test_compare_needs_a_llama_binary(self) -> None:
        instruments = self.make()
        prompt_id = instruments.dolly_prompt()["prompt_id"]
        with self.assertRaises(RuntimeError):
            instruments.compare({"prompt_id": prompt_id, "configs": {"a": "x"}})

    def test_compare_reports_each_condition_on_its_own(self) -> None:
        instruments = self.make(llama_binary="llama-server-that-does-not-exist")
        prompt_id = instruments.dolly_prompt()["prompt_id"]
        result = instruments.compare({
            "prompt_id": prompt_id,
            "configs": {
                "malformed": "not-a-key",
                "no_file": "qwen2.5-7b_t128_b1_F16_g0_c4",
                "no_binary": "qwen2.5-0.5b_t128_b1_Q4_K_M_g0_c4",
            },
        })
        rows = {row["condition"]: row for row in result["rows"]}
        self.assertIn("malformed", rows["malformed"]["error"])
        self.assertIn("no GGUF", rows["no_file"]["error"])
        # The file exists, so launching is attempted -- and fails cleanly.
        self.assertIn("BackendError", rows["no_binary"]["error"])
        self.assertEqual(result["prompt"]["prompt_id"], prompt_id)

    def test_compare_rejects_unknown_prompt(self) -> None:
        instruments = self.make(llama_binary="x")
        with self.assertRaises(ValueError):
            instruments.compare({"prompt_id": "dolly-99999", "configs": {"a": "b"}})


if __name__ == "__main__":
    unittest.main()
