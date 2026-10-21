"""Configuration vector, encoding, and space construction."""

from __future__ import annotations

import unittest

from bopis import config_space as cs
from bopis.config_space import ALL_LAYERS, Config


class TestDomains(unittest.TestCase):
    def test_documented_values(self) -> None:
        """Table 3.2's explored values, verbatim."""
        self.assertEqual(cs.T_VALUES, (128, 256, 512, 1024))
        self.assertEqual(cs.B_VALUES, (1, 2, 4, 8))
        # A-40: F32 dropped, and the model ladder added.
        self.assertEqual(cs.P_VALUES, ("F16", "Q8_0", "Q4_K_M"))
        self.assertEqual(
            cs.M_VALUES, ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")
        )
        self.assertEqual(cs.G_VALUES, (0, 14, 28, ALL_LAYERS))
        self.assertEqual(cs.C_VALUES, (2, 4, 8))

    def test_context_size_is_outside_the_search_space(self) -> None:
        """Amendment A-1: ctx is fixed, t means n_predict."""
        self.assertEqual(cs.FIXED_CTX_SIZE, 2048)
        self.assertNotIn(cs.FIXED_CTX_SIZE, cs.T_VALUES)

    def test_bits_per_weight_ordering(self) -> None:
        bpw = cs.BITS_PER_WEIGHT
        self.assertEqual(bpw["F32"], 32.0)
        self.assertEqual(bpw["F16"], 16.0)
        self.assertGreater(bpw["Q8_0"], bpw["Q4_K_M"])
        self.assertLess(bpw["Q8_0"], bpw["F16"])
        # Every searched variant must have a bit-width, or encoding raises. F32
        # stays in the table for reporting only (A-40).
        self.assertTrue(set(cs.P_VALUES) <= set(bpw))
        self.assertNotIn("F32", cs.P_VALUES)

    def test_full_space_is_2304(self) -> None:
        """4 models x 4 x 4 x 3 x 4 x 3 = 2304, still exhaustively searchable."""
        self.assertEqual(len(cs.full_space()), 2304)
        self.assertEqual(len(set(cs.full_space())), 2304)

    def test_model_ladder_is_one_family_smallest_first(self) -> None:
        params = [cs.MODELS[m].n_params for m in cs.M_VALUES]
        self.assertEqual(params, sorted(params))
        self.assertTrue(all(m.startswith("qwen2.5-") for m in cs.M_VALUES))
        self.assertIn(cs.DEFAULT_M, cs.M_VALUES)


class TestConfig(unittest.TestCase):
    def test_all_layers_label(self) -> None:
        self.assertEqual(Config(128, 1, "F32", ALL_LAYERS, 2).g_label, "All")
        self.assertEqual(Config(128, 1, "F32", 14, 2).g_label, "14")

    def test_key_is_filesystem_safe(self) -> None:
        key = Config(1024, 8, "Q4_K_M", ALL_LAYERS, 8).key()
        for bad in "/\\:*?\"<>| ":
            self.assertNotIn(bad, key)

    def test_key_is_unique_across_the_space(self) -> None:
        keys = {cfg.key() for cfg in cs.full_space()}
        self.assertEqual(len(keys), 2304)

    def test_key_carries_the_model(self) -> None:
        cfg = Config(128, 1, "Q4_K_M", 0, 4, m="qwen2.5-3b")
        self.assertTrue(cfg.key().startswith("qwen2.5-3b_"))
        # Chapter 3's positional form still builds the default model.
        self.assertEqual(Config(128, 1, "Q4_K_M", 0, 4).m, cs.DEFAULT_M)

    def test_resolved_gpu_layers(self) -> None:
        self.assertEqual(Config(128, 1, "F32", ALL_LAYERS, 2).resolved_gpu_layers(32), 32)
        self.assertEqual(Config(128, 1, "F32", 14, 2).resolved_gpu_layers(32), 14)
        # A request beyond the model's depth is capped, not an error.
        self.assertEqual(Config(128, 1, "F32", 28, 2).resolved_gpu_layers(20), 20)

    def test_gpu_fraction(self) -> None:
        self.assertEqual(Config(128, 1, "F32", 0, 2).gpu_fraction(32), 0.0)
        self.assertEqual(Config(128, 1, "F32", ALL_LAYERS, 2).gpu_fraction(32), 1.0)
        self.assertAlmostEqual(
            Config(128, 1, "F32", 16, 2).gpu_fraction(32), 0.5, places=12
        )

    def test_gpu_fraction_handles_zero_layers(self) -> None:
        self.assertEqual(Config(128, 1, "F32", ALL_LAYERS, 2).gpu_fraction(0), 0.0)

    def test_as_row_keys_match_schema(self) -> None:
        from bopis.schemas import CONFIG_COLUMNS

        row = Config(256, 2, "F16", 14, 4).as_row()
        self.assertEqual(tuple(row.keys()), CONFIG_COLUMNS)

    def test_default_config(self) -> None:
        default = cs.default_config(8)
        self.assertEqual(default.p, "F16")  # A-40: F32 is no longer searched
        self.assertEqual(default.m, cs.DEFAULT_M)
        self.assertEqual(default.b, 1)
        self.assertEqual(default.g, ALL_LAYERS)
        self.assertEqual(default.c, 8)

    def test_default_config_respects_small_core_count(self) -> None:
        # A 4-core host cannot use 8 threads.
        self.assertEqual(cs.default_config(4).c, 4)
        self.assertEqual(cs.default_config(2).c, 2)
        # Below the smallest explored value, fall back to it rather than crash.
        self.assertEqual(cs.default_config(1).c, 2)


class TestEncoding(unittest.TestCase):
    def test_produces_six_dimensions(self) -> None:
        self.assertEqual(len(cs.encode(Config(128, 1, "F16", 0, 2))), 6)
        self.assertEqual(len(cs.ENCODED_DIMS), 6)

    def test_stays_inside_unit_hypercube(self) -> None:
        for cfg in cs.full_space():
            for value in cs.encode(cfg):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_corners_map_to_zero_and_one(self) -> None:
        # Smallest t/b/c and lowest bit-width map to 0 on their axes.
        low = cs.encode(Config(128, 1, "Q4_K_M", 0, 2, m="qwen2.5-0.5b"))
        self.assertAlmostEqual(low[0], 0.0, places=12)  # t
        self.assertAlmostEqual(low[1], 0.0, places=12)  # b
        self.assertAlmostEqual(low[2], 0.0, places=12)  # bits per weight
        self.assertAlmostEqual(low[3], 0.0, places=12)  # gpu fraction
        self.assertAlmostEqual(low[4], 0.0, places=12)  # c
        self.assertAlmostEqual(low[5], 0.0, places=12)  # model size

        high = cs.encode(Config(1024, 8, "F16", ALL_LAYERS, 8, m="qwen2.5-7b"))
        for value in high:
            self.assertAlmostEqual(value, 1.0, places=12)

    def test_is_monotone_in_each_ordinal_parameter(self) -> None:
        base = dict(t=256, b=2, p="F16", g=14, c=4)

        def encoded(**overrides):
            return cs.encode(Config(**{**base, **overrides}))

        self.assertLess(encoded(t=128)[0], encoded(t=1024)[0])
        self.assertLess(encoded(b=1)[1], encoded(b=8)[1])
        self.assertLess(encoded(c=2)[4], encoded(c=8)[4])
        self.assertLess(encoded(g=0)[3], encoded(g=ALL_LAYERS)[3])

    def test_precision_ordering_is_preserved(self) -> None:
        """The categorical variant is encoded on an ordinal bit-width scale."""
        values = [
            cs.encode(Config(256, 1, variant, 14, 4))[2]
            for variant in ("Q4_K_M", "Q8_0", "F16")
        ]
        self.assertEqual(values, sorted(values))

    def test_is_deterministic(self) -> None:
        cfg = Config(512, 4, "Q8_0", 28, 4)
        self.assertEqual(cs.encode(cfg), cs.encode(cfg))

    def test_model_size_is_ordinal(self) -> None:
        values = [cs.encode(Config(256, 1, "F16", 0, 4, m=m))[5] for m in cs.M_VALUES]
        self.assertEqual(values, sorted(values))

    def test_distinct_placements_encode_distinctly(self) -> None:
        """Only configurations that are physically the same may collide.

        ``g`` is encoded as the fraction of *that model's* layers, so g=28 on a
        28-layer model is g=All -- the same placement, correctly encoded the
        same. Those duplicates are what feasible_space's DUPLICATE rule
        removes; every other pair must encode distinctly.
        """
        def placement(cfg):
            return cfg._replace(g=cfg.resolved_gpu_layers(cfg.n_layers))

        distinct = {placement(cfg): cfg for cfg in cs.full_space()}.values()
        encodings = {tuple(cs.encode(cfg)) for cfg in distinct}
        self.assertEqual(len(encodings), len(distinct))


class TestBuildSpace(unittest.TestCase):
    def test_restricting_a_domain_shrinks_the_product(self) -> None:
        space = cs.build_space(p_values=["Q8_0", "Q4_K_M"], g_values=[0, 14])
        self.assertEqual(len(space), 4 * 4 * 4 * 2 * 2 * 3)
        self.assertEqual({cfg.p for cfg in space}, {"Q8_0", "Q4_K_M"})
        self.assertEqual({cfg.g for cfg in space}, {0, 14})

    def test_precision_order_is_canonical_not_alphabetical(self) -> None:
        space = cs.build_space(p_values=["Q4_K_M", "F16"])
        seen = [cfg.p for cfg in space]
        self.assertLess(seen.index("F16"), seen.index("Q4_K_M"))

    def test_all_layers_sorts_last(self) -> None:
        space = cs.build_space(
            t_values=[128], b_values=[1], p_values=["F16"], c_values=[2],
            m_values=[cs.DEFAULT_M],
        )
        self.assertEqual([cfg.g for cfg in space], [0, 14, 28, ALL_LAYERS])

    def test_index_of(self) -> None:
        space = cs.full_space()
        target = space[42]
        self.assertEqual(cs.index_of(space, target), 42)
        self.assertEqual(cs.index_of(space, Config(1, 1, "F32", 0, 1)), -1)


if __name__ == "__main__":
    unittest.main()
