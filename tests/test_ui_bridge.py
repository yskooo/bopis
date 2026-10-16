"""The chat UI bridge: energy labelling and retrieval-based scoring.

These tests exist because two client-visible contracts live here and neither
was asserted before:

* an energy window measured by NVML must be reported as **measured**. It used
  to be pinned to RAPL alone, so on a host whose GPU reports energy but has no
  hardware-monitor feed the UI showed a real measurement as "estimated (Mode C)"
  -- the exact misrepresentation the estimator's labelling rules forbid, just in
  the opposite direction.
* a prompt with no Dolly reference (free chat) must still be scorable, with the
  retrieved match's confidence surfaced so a low-quality match is flagged rather
  than silently used.

Hardware is not touched: the GPU method is set directly and the scorer is a
stub, so these run identically on a machine with no NVIDIA GPU and no
LibreHardwareMonitor.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from bopis.monitor.nvml import EnergyMethod
from bopis.monitor.sampler import Window
from bopis.quality.bertscore import QualityScore
from bopis.ui_server import Instruments

_POOL = [
    ("how do I bake a chocolate cake", "Mix flour, sugar, cocoa, eggs, then bake.", "dolly"),
    ("what is the capital of France", "The capital of France is Paris.", "dolly"),
]


def _retriever(pairs=_POOL):
    from bopis.retrieval import ReferenceRetriever

    tmp = tempfile.mkdtemp()
    pool_path = os.path.join(tmp, "pool.json")
    with open(pool_path, "w", encoding="utf-8") as handle:
        json.dump(
            [{"question": q, "answer": a, "source": s} for q, a, s in pairs],
            handle,
        )
    retriever = ReferenceRetriever(
        pool_path=pool_path, use_sentence_transformers=False
    )
    retriever.load_pool()
    return retriever


def _instruments(**kwargs) -> Instruments:
    # data_dir has no Dolly file (so no prompt pool is loaded) and hwmon_url
    # points at a closed port, so the CPU-package source stays unavailable.
    kwargs.setdefault("data_dir", tempfile.mkdtemp())
    kwargs.setdefault("idle_seconds", 0.0)
    kwargs.setdefault("hwmon_url", "http://127.0.0.1:1/data.json")
    return Instruments(**kwargs)


class _StubScorer:
    """Stands in for BERTScore so no torch model is loaded."""

    def score(self, candidates, references):
        return [
            QualityScore(
                f1=0.8,
                precision=0.8,
                recall=0.8,
                scorer="stub",
                baseline_rescaled=False,
            )
            for _ in candidates
        ]

    def describe(self):
        return {"model": "stub-model"}


def _window(method: str, joules: float = 10.0) -> Window:
    return Window(
        duration_s=2.0,
        n_samples=10,
        energy_method=method,
        energy_j=joules,
    )


class TestEnergyLabelling(unittest.TestCase):
    def test_nvml_window_is_reported_as_measured(self) -> None:
        instruments = _instruments()
        payload = instruments._energy_summary(
            _window(EnergyMethod.NVML_ENERGY_COUNTER), None
        )
        self.assertTrue(
            payload["measured"],
            "an NVML measurement must not be reported as an estimate",
        )
        self.assertTrue(payload["gpu_measured"])
        self.assertEqual(payload["scope"], "gpu_nvml")
        self.assertIn("NVML", payload["caveat"])

    def test_combined_nvml_and_rapl_window(self) -> None:
        instruments = _instruments()
        payload = instruments._energy_summary(
            _window(
                f"{EnergyMethod.NVML_ENERGY_COUNTER}"
                f"+{EnergyMethod.RAPL_HWMON_POWER_INTEGRATION}"
            ),
            None,
        )
        self.assertTrue(payload["measured"])
        self.assertTrue(payload["gpu_measured"])
        self.assertTrue(payload["rapl_measured"])
        self.assertEqual(payload["scope"], "gpu_nvml+cpu_package_rapl")

    def test_rapl_window_is_reported_as_measured(self) -> None:
        instruments = _instruments()
        payload = instruments._energy_summary(
            _window(EnergyMethod.RAPL_HWMON_POWER_INTEGRATION), None
        )
        self.assertTrue(payload["measured"])
        self.assertEqual(payload["scope"], "cpu_package_rapl")

    def test_estimator_window_keeps_the_estimate_label(self) -> None:
        instruments = _instruments()
        payload = instruments._energy_summary(
            _window(EnergyMethod.RESOURCE_ALLOCATION_ESTIMATE), None
        )
        self.assertFalse(payload["measured"])
        self.assertEqual(payload["scope"], "estimated_resource_allocation")
        # The estimate must still carry its caveat, never a measurement's.
        self.assertIn("estimate", payload["caveat"].lower())

    def test_unavailable_window_is_neither_measured_nor_estimated(self) -> None:
        instruments = _instruments()
        payload = instruments._energy_summary(
            Window(duration_s=1.0, n_samples=0, energy_method=EnergyMethod.UNAVAILABLE),
            None,
        )
        self.assertFalse(payload["measured"])
        self.assertEqual(payload["scope"], "unavailable")


class TestMeasuringFlag(unittest.TestCase):
    def test_measuring_true_when_the_gpu_reports_energy(self) -> None:
        instruments = _instruments()
        instruments.gpu_method = EnergyMethod.NVML_ENERGY_COUNTER
        self.assertTrue(instruments.gpu_measuring)
        self.assertTrue(instruments.measuring)

    def test_measuring_false_with_no_sensor_at_all(self) -> None:
        instruments = _instruments()
        instruments.gpu_method = EnergyMethod.UNAVAILABLE
        instruments.source = None
        self.assertFalse(instruments.gpu_measuring)
        self.assertFalse(instruments.measuring)


class TestRetrievalScoring(unittest.TestCase):
    def test_empty_reference_is_filled_from_the_pool(self) -> None:
        instruments = _instruments()
        instruments._retriever = _retriever()
        instruments.retriever_error = None
        instruments._scorer = _StubScorer()

        payload = instruments.score("chocolate cake baking steps", "")
        self.assertIn("reference_match", payload)
        match = payload["reference_match"]
        self.assertEqual(match["matched_question"], _POOL[0][0])
        # Exposed so the UI applies the 0.5 honesty rule.
        self.assertGreater(match["similarity"], 0.0)
        self.assertIn("low_confidence", match)
        self.assertIsInstance(match["low_confidence"], bool)
        self.assertIn("reference", payload)

    def test_prompt_drives_the_retrieval_not_the_reply(self) -> None:
        """Free chat matches the user's question, not the assistant's reply.

        Matching the reply would score it against an unrelated pooled answer --
        the bug this guards against. The reply here shares nothing with the
        pool; only *prompt* points at the France/capital pair.
        """
        instruments = _instruments()
        instruments._retriever = _retriever()
        instruments.retriever_error = None
        instruments._scorer = _StubScorer()

        payload = instruments.score(
            "I am a helpful assistant.",
            "",
            prompt="what is the capital of France",
        )
        self.assertEqual(payload["reference_match"]["matched_question"], _POOL[1][0])
        self.assertEqual(payload["reference"], _POOL[1][1])

    def test_reply_only_call_still_works_without_a_prompt(self) -> None:
        """Callers that predate the prompt argument keep matching the candidate."""
        instruments = _instruments()
        instruments._retriever = _retriever()
        instruments.retriever_error = None
        instruments._scorer = _StubScorer()
        payload = instruments.score("how do I bake a chocolate cake", "")
        self.assertEqual(payload["reference_match"]["matched_question"], _POOL[0][0])

    def test_no_pool_and_no_reference_is_an_error(self) -> None:
        instruments = _instruments()
        instruments._retriever = None
        instruments.retriever_error = "pool not built"
        instruments._scorer = _StubScorer()
        with self.assertRaises(ValueError):
            instruments.score("anything", "")

    def test_no_match_reports_the_backend_accept_threshold(self) -> None:
        """The 400 message must name 0.10 for TF-IDF, not the 0.5 flag."""
        instruments = _instruments()
        instruments._retriever = _retriever()
        instruments.retriever_error = None
        instruments._scorer = _StubScorer()
        with self.assertRaises(ValueError) as caught:
            instruments.score("zzz qqq vvv www jjj kkk", "")
        self.assertIn("0.10", str(caught.exception))
        self.assertNotIn("0.50", str(caught.exception))

    def test_weak_match_is_flagged_low_confidence_but_still_scored(self) -> None:
        # A question that shares only a little vocabulary with the pool clears
        # the 0.1 accept threshold but lands under the 0.5 flag.
        weak = [("a very long question about many unrelated exotic topics indeed", "an answer", "dolly")]
        instruments = _instruments()
        instruments._retriever = _retriever(weak)
        instruments.retriever_error = None
        instruments._scorer = _StubScorer()
        payload = instruments.score("topics", "")
        match = payload["reference_match"]
        self.assertGreaterEqual(match["similarity"], 0.1)
        self.assertLess(match["similarity"], 0.5)
        self.assertTrue(match["low_confidence"])

    def test_dolly_reference_still_scores_directly(self) -> None:
        instruments = _instruments()
        instruments._scorer = _StubScorer()
        payload = instruments.score("candidate", "a human reference")
        self.assertNotIn("reference_match", payload)
        self.assertEqual(payload["reference"], "a human reference")


class TestStudyEnergyMode(unittest.TestCase):
    def test_real_study_prefers_auto_when_the_gpu_measures(self) -> None:
        instruments = _instruments(llama_binary="llama-server.exe")
        instruments.model_paths = {"q4": "model.gguf"}
        instruments.gpu_method = EnergyMethod.NVML_ENERGY_COUNTER
        command = instruments.study_command({"preset": "real"})
        mode = command[command.index("--energy-mode") + 1]
        self.assertEqual(mode, "auto")

    def test_real_study_falls_back_without_a_gpu_sensor(self) -> None:
        instruments = _instruments(llama_binary="llama-server.exe")
        instruments.model_paths = {"q4": "model.gguf"}
        instruments.gpu_method = EnergyMethod.UNAVAILABLE
        instruments.source = None
        command = instruments.study_command({"preset": "real"})
        mode = command[command.index("--energy-mode") + 1]
        self.assertEqual(mode, "resource-estimate")


if __name__ == "__main__":
    unittest.main()
