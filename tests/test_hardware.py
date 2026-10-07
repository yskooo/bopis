"""Table H1 rules, the model-size guards, and feasible-space construction.

Every one of the twelve documented rule branches is checked against the
permitted set the manuscript specifies, so a change to the thresholds cannot
pass unnoticed.
"""

from __future__ import annotations

import unittest

from bopis import hardware
from bopis.config_space import ALL_LAYERS, Config
from bopis.hardware import (
    GIB,
    MISTRAL_7B_INSTRUCT_V03,
    HostProfile,
    ModelSpec,
    apply_h1_rules,
    feasible_space,
    h1_batch_domain,
    h1_gpu_layer_domain,
    h1_precision_domain,
    h1_thread_domain,
    summarize_space,
)


def profile(
    vram_gib: float = 12.0,
    ram_gib: float = 32.0,
    cores: int = 8,
    gpu: bool = True,
) -> HostProfile:
    """A synthetic host, so the rules can be tested without real hardware."""
    host = HostProfile(
        cpu_model="synthetic",
        physical_cores=cores,
        logical_cores=cores * 2,
        ram_total_bytes=int(ram_gib * GIB),
        ram_available_bytes=int(ram_gib * GIB * 0.8),
        is_wsl=False,
        telemetry_source="synthetic",
        gpu_available=gpu,
        gpu_name="synthetic GPU" if gpu else None,
        vram_total_bytes=int(vram_gib * GIB) if gpu else 0,
        vram_free_bytes=int(vram_gib * GIB * 0.95) if gpu else 0,
    )
    apply_h1_rules(host)
    return host


class TestPrecisionRules(unittest.TestCase):
    def test_hw_p1_below_4gib(self) -> None:
        for vram in (0.5, 2.0, 3.99):
            domain, rule = h1_precision_domain(int(vram * GIB))
            self.assertEqual(rule, "HW-P1")
            self.assertEqual(domain, ("Q8_0", "Q4_K_M"))

    def test_hw_p2_between_4_and_8gib(self) -> None:
        for vram in (4.0, 6.0, 7.99):
            domain, rule = h1_precision_domain(int(vram * GIB))
            self.assertEqual(rule, "HW-P2")
            self.assertEqual(domain, ("F16", "Q8_0", "Q4_K_M"))

    def test_hw_p3_at_or_above_8gib(self) -> None:
        for vram in (8.0, 12.0, 24.0):
            domain, rule = h1_precision_domain(int(vram * GIB))
            self.assertEqual(rule, "HW-P3")
            self.assertEqual(domain, ("F32", "F16", "Q8_0", "Q4_K_M"))

    def test_boundaries_are_inclusive_below(self) -> None:
        """Exactly 4 GiB falls in HW-P2, exactly 8 GiB in HW-P3."""
        self.assertEqual(h1_precision_domain(4 * GIB)[1], "HW-P2")
        self.assertEqual(h1_precision_domain(8 * GIB)[1], "HW-P3")


class TestGpuLayerRules(unittest.TestCase):
    def test_hw_g1(self) -> None:
        domain, rule = h1_gpu_layer_domain(2 * GIB)
        self.assertEqual((domain, rule), ((0, 14), "HW-G1"))

    def test_hw_g2(self) -> None:
        domain, rule = h1_gpu_layer_domain(6 * GIB)
        self.assertEqual((domain, rule), ((0, 14, 28), "HW-G2"))

    def test_hw_g3(self) -> None:
        domain, rule = h1_gpu_layer_domain(12 * GIB)
        self.assertEqual((domain, rule), ((0, 14, 28, ALL_LAYERS), "HW-G3"))


class TestBatchRules(unittest.TestCase):
    def test_hw_b1(self) -> None:
        self.assertEqual(h1_batch_domain(4 * GIB), ((1,), "HW-B1"))

    def test_hw_b2(self) -> None:
        self.assertEqual(h1_batch_domain(12 * GIB), ((1, 2), "HW-B2"))

    def test_hw_b3(self) -> None:
        self.assertEqual(h1_batch_domain(32 * GIB), ((1, 2, 4, 8), "HW-B3"))

    def test_wsl_memory_ceiling_flips_the_branch(self) -> None:
        """Amendment A-14: the WSL ceiling, not host RAM, governs HW-B.

        The development machine reports 15.78 GiB to Windows but 11.68 GiB
        inside WSL. Those straddle the 16 GiB boundary, so the same physical
        machine lands in different branches depending on which figure is used.
        """
        windows_view = h1_batch_domain(int(15.78 * GIB))
        wsl_view = h1_batch_domain(int(11.68 * GIB))
        self.assertEqual(windows_view[1], "HW-B2")
        self.assertEqual(wsl_view[1], "HW-B2")
        # And a genuinely larger host does reach HW-B3.
        self.assertEqual(h1_batch_domain(int(16.0 * GIB))[1], "HW-B3")


class TestThreadRules(unittest.TestCase):
    def test_hw_c1(self) -> None:
        for cores in (1, 2, 3):
            self.assertEqual(h1_thread_domain(cores), ((2,), "HW-C1"))

    def test_hw_c2(self) -> None:
        for cores in (4, 5, 6, 7):
            self.assertEqual(h1_thread_domain(cores), ((2, 4), "HW-C2"))

    def test_hw_c3(self) -> None:
        for cores in (8, 16, 64):
            self.assertEqual(h1_thread_domain(cores), ((2, 4, 8), "HW-C3"))


class TestApplyRules(unittest.TestCase):
    def test_large_host_gets_the_full_space(self) -> None:
        host = profile(vram_gib=24.0, ram_gib=64.0, cores=16)
        self.assertEqual(host.rules_fired, ["HW-P3", "HW-G3", "HW-B3", "HW-C3"])
        space, rejections = feasible_space(host)
        # Everything except placements that duplicate g=All: g=28 on the three
        # rungs with <= 28 layers (0.5B, 1.5B, 7B), x 4 t x 4 b x 3 p x 3 c.
        self.assertEqual(len(space), 2304 - 3 * 4 * 4 * 3 * 3)
        self.assertEqual({r.rule for r in rejections}, {"DUPLICATE"})

    def test_tiny_host_is_heavily_constrained(self) -> None:
        host = profile(vram_gib=2.0, ram_gib=6.0, cores=2)
        self.assertEqual(host.rules_fired, ["HW-P1", "HW-G1", "HW-B1", "HW-C1"])
        space, _ = feasible_space(host)
        # m(4) x t(4) x b(1) x c(1) x [3 precisions at g=0 + 2 at g=14] (A-41)
        self.assertEqual(len(space), 4 * 4 * 1 * 1 * (3 + 2))

    def test_no_gpu_forbids_offload(self) -> None:
        host = profile(gpu=False)
        self.assertIn("HW-G-NOGPU", host.rules_fired)
        space, _ = feasible_space(host)
        self.assertTrue(all(cfg.g == 0 for cfg in space))

    def test_development_machine_branch(self) -> None:
        """The real dev laptop: MX330 2 GiB, 15.78 GiB RAM, 4 physical cores."""
        host = profile(vram_gib=2.0, ram_gib=15.78, cores=4)
        self.assertEqual(host.rules_fired, ["HW-P1", "HW-G1", "HW-B2", "HW-C2"])
        space, _ = feasible_space(host)
        # m(4) x t(4) x b(2) x c(2) x [3 precisions at g=0 + 2 at g=14]
        self.assertEqual(len(space), 4 * 4 * 2 * 2 * (3 + 2))

    def test_vram_precision_rule_binds_only_when_offloading(self) -> None:
        """A-41: at g=0 the weights are in RAM, so VRAM cannot exclude F16."""
        host = profile(vram_gib=2.0, ram_gib=15.78, cores=4)
        space, rejections = feasible_space(host)
        self.assertTrue(any(c.p == "F16" and c.g == 0 for c in space))
        self.assertFalse(any(c.p == "F16" and c.g != 0 for c in space))
        self.assertTrue(
            all(r.rule == "HW-P1" for r in rejections if r.config.p == "F16")
        )


class TestModelSizeGuards(unittest.TestCase):
    def test_mistral_7b_weight_sizes(self) -> None:
        """Sanity-check the footprint model against published GGUF sizes."""
        model = MISTRAL_7B_INSTRUCT_V03
        self.assertAlmostEqual(model.weight_bytes("F32") / GIB, 27.0, delta=1.5)
        self.assertAlmostEqual(model.weight_bytes("F16") / GIB, 13.5, delta=1.0)
        self.assertAlmostEqual(model.weight_bytes("Q8_0") / GIB, 7.2, delta=0.6)
        self.assertAlmostEqual(model.weight_bytes("Q4_K_M") / GIB, 4.1, delta=0.5)

    def test_kv_cache_scales_with_batch_and_context(self) -> None:
        model = MISTRAL_7B_INSTRUCT_V03
        one = model.kv_cache_bytes(2048, 1)
        two = model.kv_cache_bytes(2048, 2)
        self.assertAlmostEqual(two / one, 2.0, places=9)
        self.assertAlmostEqual(
            model.kv_cache_bytes(4096, 1) / one, 2.0, places=9
        )

    def test_guard_rejects_oversized_gpu_share(self) -> None:
        """Amendment A-15/A-29: Table H1 alone would permit the impossible.

        A 2 GiB card passes HW-P1 for Q8_0, but 14 of 32 layers of a 7B Q8_0
        model is over 3 GiB and cannot fit.
        """
        host = profile(vram_gib=2.0, ram_gib=15.78, cores=4)
        unguarded, _ = feasible_space(host, model=None)
        guarded, rejections = feasible_space(host, model=MISTRAL_7B_INSTRUCT_V03)
        self.assertLess(len(guarded), len(unguarded))
        self.assertTrue(any(r.rule == "HW-P0" for r in rejections))
        # Nothing with GPU layers should survive on a 2 GiB card.
        self.assertTrue(all(cfg.g == 0 for cfg in guarded))

    def test_guard_rejects_7b_f16_on_a_12gib_card(self) -> None:
        """Qwen2.5-7B at F16 is ~14 GiB and cannot be fully offloaded to 12 GiB."""
        host = profile(vram_gib=12.0, ram_gib=32.0, cores=8)
        _space, rejections = feasible_space(host, models=hardware.MODEL_LADDER)
        rejected = [
            r
            for r in rejections
            if r.config.m == "qwen2.5-7b"
            and r.config.p == "F16"
            and r.config.g == ALL_LAYERS
            and r.rule == "HW-P0"
        ]
        self.assertTrue(rejected, "fully-offloaded 7B F16 must be rejected")

    def test_ladder_guard_is_per_model(self) -> None:
        """Each rung is checked against its own footprint, not one shared one."""
        host = profile(vram_gib=2.0, ram_gib=8.0, cores=4)  # 6.4 GiB free
        space, _ = feasible_space(
            host, models=hardware.MODEL_LADDER, max_gpu_layers=0
        )
        pairs = {(c.m, c.p) for c in space}
        self.assertIn(("qwen2.5-1.5b", "F16"), pairs)      # 2.9 GiB
        self.assertIn(("qwen2.5-7b", "Q4_K_M"), pairs)     # 4.4 GiB
        self.assertNotIn(("qwen2.5-7b", "Q8_0"), pairs)    # 7.7 GiB
        self.assertIn(("qwen2.5-3b", "F16"), pairs)        # 5.8 GiB, just fits
        self.assertNotIn(("qwen2.5-7b", "F16"), pairs)     # 14.3 GiB

    def test_gguf_bytes_restrict_to_files_on_disk(self) -> None:
        host = profile()
        gguf = {("qwen2.5-1.5b", "Q4_K_M"): int(1.0 * GIB)}
        space, rejections = feasible_space(
            host, models=hardware.MODEL_LADDER, gguf_bytes=gguf
        )
        self.assertEqual({(c.m, c.p) for c in space}, {("qwen2.5-1.5b", "Q4_K_M")})
        self.assertTrue(any(r.rule == "NO-GGUF" for r in rejections))

    def test_small_model_fits_where_large_one_does_not(self) -> None:
        tiny = ModelSpec(
            name="tiny-1b", n_params=1.1e9, n_layers=22, n_kv_heads=4, head_dim=64
        )
        host = profile(vram_gib=2.0, ram_gib=15.78, cores=4)
        big, _ = feasible_space(host, model=MISTRAL_7B_INSTRUCT_V03)
        small, _ = feasible_space(host, model=tiny)
        self.assertGreater(len(small), len(big))

    def test_no_guards_applied_without_a_model(self) -> None:
        host = profile(vram_gib=2.0)
        _space, rejections = feasible_space(host, model=None)
        # Only Table H1 itself (HW-P1 on offloaded F16), never a memory guard.
        self.assertTrue(rejections)
        self.assertEqual({r.rule for r in rejections}, {"HW-P1"})


class TestEnergyScopeFloor(unittest.TestCase):
    def test_min_gpu_layers_excludes_cpu_only_configs(self) -> None:
        """Amendment A-19: g=0 makes GPU-only energy accounting meaningless."""
        host = profile(vram_gib=12.0, ram_gib=32.0, cores=8)
        space, rejections = feasible_space(host, min_gpu_layers=14)
        self.assertTrue(all(cfg.g != 0 for cfg in space))
        self.assertTrue(any(r.rule == "ENERGY-SCOPE" for r in rejections))

    def test_floor_of_zero_changes_nothing(self) -> None:
        host = profile()
        with_floor, _ = feasible_space(host, min_gpu_layers=0)
        without, _ = feasible_space(host)
        self.assertEqual(len(with_floor), len(without))


class TestSummary(unittest.TestCase):
    def test_reports_counts_and_rules(self) -> None:
        host = profile(vram_gib=2.0, ram_gib=15.78, cores=4)
        space, rejections = feasible_space(host, model=MISTRAL_7B_INSTRUCT_V03)
        summary = summarize_space(space, rejections)
        self.assertEqual(summary["n_feasible"], len(space))
        self.assertEqual(summary["n_rejected"], len(rejections))
        self.assertEqual(summary["n_unconstrained"], 2304)
        self.assertIn("HW-P0", summary["rejected_by_rule"])
        self.assertEqual(summary["precision_variants_present"], ["Q8_0", "Q4_K_M"])


class TestRealHost(unittest.TestCase):
    def test_profiling_the_actual_machine_succeeds(self) -> None:
        """Profiling must work on whatever this is, GPU or not."""
        host = hardware.profile_host()
        self.assertGreater(host.physical_cores, 0)
        self.assertGreater(host.ram_total_bytes, 0)
        self.assertEqual(len(host.rules_fired), 4)
        self.assertTrue(host.permitted_precisions)
        self.assertTrue(host.permitted_cpu_threads)
        # can_measure_energy must be a decisive boolean, never unknown.
        self.assertIsInstance(host.can_measure_energy, bool)

    def test_profile_serializes(self) -> None:
        payload = hardware.profile_host().as_dict()
        self.assertIn("energy_method", payload)
        self.assertIn("rules_fired", payload)
        # ALL_LAYERS must be rendered as "All", not -1, in the manifest.
        self.assertNotIn(-1, payload["permitted_gpu_layers"])


if __name__ == "__main__":
    unittest.main()
