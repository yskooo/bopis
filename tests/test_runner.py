"""Regression tests for the unoptimized default reference."""

from __future__ import annotations

import unittest

from bopis import config_space as cs
from bopis.hardware import GIB, HostProfile, apply_h1_rules, feasible_space
from bopis.runner import RunSettings, Study


def _host(vram_gib: float, physical_cores: int = 6) -> HostProfile:
    host = HostProfile(
        cpu_model="synthetic",
        physical_cores=physical_cores,
        logical_cores=physical_cores * 2,
        ram_total_bytes=16 * GIB,
        ram_available_bytes=12 * GIB,
        is_wsl=False,
        telemetry_source="synthetic",
        gpu_available=True,
        gpu_name="synthetic",
        vram_total_bytes=int(vram_gib * GIB),
        vram_free_bytes=int(vram_gib * GIB),
    )
    apply_h1_rules(host)
    return host


def _study(profile: HostProfile) -> Study:
    space, _ = feasible_space(profile)
    return Study(
        run_dir=None,
        settings=RunSettings(backend="llama-server"),
        profile=profile,
        space=space,
        samples=None,
        measurer=None,
    )


class ReferenceDefaultTests(unittest.TestCase):
    def test_explicit_all_layer_count_is_the_default(self) -> None:
        study = _study(_host(6))
        default = study.reference_default()

        self.assertEqual(default.g, cs.ALL_LAYERS)
        self.assertEqual(default.resolved_gpu_layers(default.n_layers or 0), 28)
        self.assertIsNone(study.default_substituted_for)

    def test_sentinel_stays_when_the_domain_lists_it(self) -> None:
        study = _study(_host(9, physical_cores=8))
        default = study.reference_default()

        self.assertEqual(default.g, cs.ALL_LAYERS)
        self.assertIsNone(study.default_substituted_for)

    def test_fallback_keeps_gpu_offload_when_all_is_not_feasible(self) -> None:
        study = _study(_host(3.5))
        default = study.reference_default()

        self.assertNotEqual(default.g, 0)
        self.assertEqual(default.resolved_gpu_layers(default.n_layers or 0), 14)
        self.assertIsNotNone(study.default_substituted_for)


if __name__ == "__main__":
    unittest.main()
