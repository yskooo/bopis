"""Pareto dominance, front extraction, hypervolume, and x* selection."""

from __future__ import annotations

import random
import unittest

from bopis.config_space import Config
from bopis.pareto import (
    Evaluation,
    Objectives,
    SelectionStatus,
    dominates,
    front_mask,
    hypervolume,
    hypervolume_monte_carlo,
    pareto_front,
    select_xstar,
)

CFG = Config(t=256, b=1, p="F16", g=14, c=4)


def ev(energy: float, speed: float, quality: float, iteration: int = 0) -> Evaluation:
    return Evaluation(CFG, Objectives(energy, speed, quality), iteration=iteration)


class TestDominance(unittest.TestCase):
    def test_strictly_better_on_all_axes_dominates(self) -> None:
        better = Objectives(10.0, 50.0, 0.9)
        worse = Objectives(20.0, 40.0, 0.8)
        self.assertTrue(dominates(better, worse))
        self.assertFalse(dominates(worse, better))

    def test_identical_does_not_dominate(self) -> None:
        """Dominance requires at least one strict inequality."""
        a = Objectives(10.0, 50.0, 0.9)
        b = Objectives(10.0, 50.0, 0.9)
        self.assertFalse(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_equal_on_two_axes_better_on_one_dominates(self) -> None:
        a = Objectives(10.0, 50.0, 0.9)
        b = Objectives(10.0, 50.0, 0.8)
        self.assertTrue(dominates(a, b))

    def test_trade_off_is_mutually_non_dominating(self) -> None:
        cheap = Objectives(10.0, 50.0, 0.70)
        accurate = Objectives(20.0, 50.0, 0.95)
        self.assertFalse(dominates(cheap, accurate))
        self.assertFalse(dominates(accurate, cheap))

    def test_energy_direction_is_minimization(self) -> None:
        low = Objectives(5.0, 10.0, 0.5)
        high = Objectives(500.0, 10.0, 0.5)
        self.assertTrue(dominates(low, high))
        self.assertFalse(dominates(high, low))

    def test_speed_and_quality_are_maximization(self) -> None:
        fast = Objectives(10.0, 99.0, 0.5)
        slow = Objectives(10.0, 1.0, 0.5)
        self.assertTrue(dominates(fast, slow))

        good = Objectives(10.0, 10.0, 0.99)
        bad = Objectives(10.0, 10.0, 0.01)
        self.assertTrue(dominates(good, bad))


class TestFront(unittest.TestCase):
    def test_matches_brute_force_on_random_sets(self) -> None:
        rng = random.Random(3)
        for _ in range(30):
            population = [
                ev(rng.uniform(20, 200), rng.uniform(5, 60), rng.uniform(0.5, 0.95))
                for _ in range(random.Random(rng.random()).randint(4, 25))
            ]
            front = pareto_front(population)
            brute = [
                candidate
                for candidate in population
                if not any(
                    dominates(other.objectives, candidate.objectives)
                    for other in population
                    if other is not candidate
                )
            ]
            self.assertEqual({id(x) for x in front}, {id(x) for x in brute})

    def test_no_front_member_is_dominated(self) -> None:
        rng = random.Random(9)
        population = [
            ev(rng.uniform(20, 200), rng.uniform(5, 60), rng.uniform(0.5, 0.95))
            for _ in range(40)
        ]
        front = pareto_front(population)
        for member in front:
            self.assertFalse(
                any(
                    dominates(other.objectives, member.objectives)
                    for other in population
                )
            )

    def test_single_point_is_its_own_front(self) -> None:
        only = ev(10.0, 20.0, 0.8)
        self.assertEqual(pareto_front([only]), [only])

    def test_empty_input(self) -> None:
        self.assertEqual(pareto_front([]), [])

    def test_mask_aligns_with_input_order(self) -> None:
        population = [ev(10, 50, 0.9), ev(20, 40, 0.8), ev(5, 60, 0.95)]
        mask = front_mask(population)
        self.assertEqual(len(mask), 3)
        # Index 2 dominates everything.
        self.assertTrue(mask[2])
        self.assertFalse(mask[1])


class TestHypervolume(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = Objectives(energy_j=200.0, tokens_per_s=10.0, quality_f1=0.60)

    def test_single_best_corner_is_one_when_normalized(self) -> None:
        """One point that is best on every axis fills the normalized cube."""
        front = [ev(20.0, 60.0, 0.95)]
        self.assertAlmostEqual(hypervolume(front, self.reference), 1.0, places=9)

    def test_point_at_reference_contributes_nothing(self) -> None:
        front = [ev(200.0, 10.0, 0.60)]
        self.assertAlmostEqual(hypervolume(front, self.reference), 0.0, places=9)

    def test_matches_monte_carlo(self) -> None:
        rng = random.Random(21)
        for trial in range(6):
            population = [
                ev(
                    rng.uniform(20, 190),
                    rng.uniform(12, 60),
                    rng.uniform(0.62, 0.95),
                )
                for _ in range(12)
            ]
            front = pareto_front(population)
            exact = hypervolume(front, self.reference)
            approx = hypervolume_monte_carlo(
                front, self.reference, samples=120_000, seed=trial
            )
            self.assertAlmostEqual(exact, approx, delta=0.01)

    def test_adding_a_dominating_point_cannot_shrink_volume(self) -> None:
        base = [ev(120.0, 30.0, 0.80)]
        grown = base + [ev(60.0, 45.0, 0.90)]
        self.assertGreaterEqual(
            hypervolume(pareto_front(grown), self.reference),
            hypervolume(base, self.reference),
        )

    def test_normalized_result_is_within_unit_range(self) -> None:
        rng = random.Random(33)
        population = [
            ev(rng.uniform(20, 190), rng.uniform(12, 60), rng.uniform(0.62, 0.95))
            for _ in range(15)
        ]
        value = hypervolume(pareto_front(population), self.reference)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_empty_front(self) -> None:
        self.assertEqual(hypervolume([], self.reference), 0.0)

    def test_unnormalized_carries_raw_units(self) -> None:
        """Raw HV is much larger than 1: exactly why it must be normalized."""
        front = [ev(20.0, 60.0, 0.95)]
        raw = hypervolume(front, self.reference, normalize=False)
        self.assertGreater(raw, 1.0)

    def test_positive_when_the_front_cannot_beat_the_reference_on_one_axis(
        self,
    ) -> None:
        """The realistic case Chapter 3's reference point breaks on.

        The unoptimized default runs F32 -- the highest-quality variant in the
        space -- so no configuration can beat it on quality. Against an unclamped
        ``r = (E_default, -S_default, -Q_default)`` the quality axis has zero
        span and HV collapses to 0 no matter how much energy was saved. These are
        real values from a run: nine front members, none reaching the default's
        0.8615 F1, alongside a 41% energy reduction.
        """
        default = Objectives(energy_j=241.32, tokens_per_s=14.59, quality_f1=0.8615)
        front = [
            ev(63.21, 28.35, 0.7934),
            ev(83.37, 18.98, 0.7945),
            ev(86.79, 22.13, 0.8198),
            ev(93.93, 28.95, 0.8128),
            ev(101.16, 29.12, 0.8245),
            ev(141.20, 22.48, 0.8507),
        ]
        self.assertTrue(
            all(e.quality_f1 < default.quality_f1 for e in front),
            "the fixture must reproduce the degenerate case",
        )
        value = hypervolume(front, default)
        self.assertGreater(
            value,
            0.0,
            "hypervolume must stay informative when the front cannot beat the "
            "reference on one axis; the reference is clamped to the nadir",
        )
        self.assertLessEqual(value, 1.0)

    def test_reference_clamping_does_not_inflate_a_dominating_front(self) -> None:
        """Where the default really is worst on every axis, nothing changes."""
        default = Objectives(energy_j=300.0, tokens_per_s=5.0, quality_f1=0.50)
        front = [ev(50.0, 40.0, 0.90)]
        # A single point beating the reference on all axes fills the cube.
        self.assertAlmostEqual(hypervolume(front, default), 1.0, places=9)


class TestSelection(unittest.TestCase):
    def setUp(self) -> None:
        # Baseline: 100 J, 20 tok/s, 0.80 F1
        self.baseline = Objectives(100.0, 20.0, 0.80)

    def test_picks_highest_eir_among_admissible(self) -> None:
        population = [
            ev(80.0, 20.0, 0.80),  # EIR 20%, SRR 100%, QRR 100%
            ev(60.0, 20.0, 0.80),  # EIR 40% -- should win
            ev(90.0, 25.0, 0.82),
        ]
        selection = select_xstar(population, self.baseline)
        self.assertEqual(selection.status, SelectionStatus.OPTIMAL)
        self.assertAlmostEqual(selection.eir_percent, 40.0, places=6)

    def test_rejects_candidate_failing_qrr(self) -> None:
        """A big energy win must not be selected if quality collapses."""
        population = [
            ev(30.0, 25.0, 0.50),  # EIR 70% but QRR 62.5% -- inadmissible
            ev(85.0, 21.0, 0.79),  # EIR 15%, QRR 98.75% -- admissible
        ]
        selection = select_xstar(population, self.baseline)
        self.assertEqual(selection.status, SelectionStatus.OPTIMAL)
        self.assertAlmostEqual(selection.qrr_percent, 98.75, places=6)

    def test_rejects_candidate_failing_srr(self) -> None:
        population = [
            ev(40.0, 5.0, 0.80),  # EIR 60% but SRR 25%
            ev(88.0, 20.0, 0.80),
        ]
        selection = select_xstar(population, self.baseline)
        self.assertGreaterEqual(selection.srr_percent, 95.0)

    def test_relaxes_qrr_when_nothing_qualifies(self) -> None:
        # QRR 96.25% -- below 98 but at or above the relaxed 95.
        population = [ev(70.0, 20.0, 0.77)]
        selection = select_xstar(population, self.baseline)
        self.assertEqual(selection.status, SelectionStatus.RELAXED_QRR)
        self.assertFalse(selection.is_clean)
        self.assertIn("relaxed", selection.notes)

    def test_falls_back_to_minimum_energy_and_says_so(self) -> None:
        # Fails every rung of the ladder.
        population = [ev(70.0, 2.0, 0.20), ev(50.0, 1.0, 0.10)]
        selection = select_xstar(population, self.baseline)
        self.assertEqual(selection.status, SelectionStatus.MIN_ENERGY_FALLBACK)
        self.assertFalse(selection.is_clean)
        self.assertIn("NOT a successful optimization", selection.notes)
        self.assertAlmostEqual(selection.evaluation.energy_j, 50.0, places=6)

    def test_empty_population(self) -> None:
        selection = select_xstar([], self.baseline)
        self.assertEqual(selection.status, SelectionStatus.EMPTY)
        self.assertIsNone(selection.config)

    def test_selected_member_is_on_the_front(self) -> None:
        rng = random.Random(41)
        population = [
            ev(rng.uniform(40, 120), rng.uniform(15, 40), rng.uniform(0.75, 0.9))
            for _ in range(20)
        ]
        selection = select_xstar(population, self.baseline)
        front_ids = {id(e) for e in pareto_front(population)}
        self.assertIn(id(selection.evaluation), front_ids)

    def test_is_clean_only_for_optimal_status(self) -> None:
        clean = select_xstar([ev(80.0, 20.0, 0.80)], self.baseline)
        self.assertTrue(clean.is_clean)
        relaxed = select_xstar([ev(70.0, 20.0, 0.77)], self.baseline)
        self.assertFalse(relaxed.is_clean)


if __name__ == "__main__":
    unittest.main()
