"""Expected Improvement, including the guard on the manuscript's sign error.

``test_prefers_lower_predicted_energy`` is the important one: it fails against
Chapter 3's printed formula and passes against the corrected minimization form.
It is the reason the correction cannot silently regress.
"""

from __future__ import annotations

import math
import unittest

from bopis.acquisition import (
    argmax_expected_improvement,
    expected_improvement,
    standard_normal_cdf,
    standard_normal_pdf,
)


class TestNormalFunctions(unittest.TestCase):
    def test_cdf_known_values(self) -> None:
        self.assertAlmostEqual(standard_normal_cdf(0.0), 0.5, places=12)
        self.assertAlmostEqual(standard_normal_cdf(1.959963985), 0.975, places=8)
        self.assertAlmostEqual(standard_normal_cdf(-1.959963985), 0.025, places=8)

    def test_cdf_is_monotonic(self) -> None:
        values = [standard_normal_cdf(z) for z in (-3, -1, 0, 1, 3)]
        self.assertEqual(values, sorted(values))

    def test_pdf_known_values(self) -> None:
        self.assertAlmostEqual(
            standard_normal_pdf(0.0), 1.0 / math.sqrt(2 * math.pi), places=12
        )
        self.assertAlmostEqual(standard_normal_pdf(1.0), 0.241970724519, places=10)

    def test_pdf_is_symmetric(self) -> None:
        for z in (0.3, 1.0, 2.5):
            self.assertAlmostEqual(
                standard_normal_pdf(z), standard_normal_pdf(-z), places=14
            )


class TestExpectedImprovement(unittest.TestCase):
    def test_prefers_lower_predicted_energy(self) -> None:
        """EI must be higher for the candidate predicted to use LESS energy.

        This encodes the direction of the objective. Chapter 3 writes the
        improvement term as ``mu(x) - f(x+)``, which -- since energy is
        minimized and ``f(x+)`` is the best (lowest) value so far -- rewards
        candidates predicted to be *worse*. Under that formula this assertion
        fails.
        """
        f_best = 100.0
        sigma = 5.0
        better = expected_improvement(mu=80.0, sigma=sigma, f_best=f_best)
        worse = expected_improvement(mu=120.0, sigma=sigma, f_best=f_best)
        self.assertGreater(
            better,
            worse,
            "EI must favour lower predicted energy; the sign of the "
            "improvement term is inverted",
        )

    def test_never_negative(self) -> None:
        for mu in (10.0, 50.0, 100.0, 500.0):
            for sigma in (0.01, 1.0, 25.0):
                self.assertGreaterEqual(
                    expected_improvement(mu, sigma, f_best=100.0), 0.0
                )

    def test_zero_when_certain(self) -> None:
        """A candidate with no posterior uncertainty offers no improvement."""
        self.assertEqual(expected_improvement(50.0, 0.0, f_best=100.0), 0.0)
        self.assertEqual(expected_improvement(150.0, 0.0, f_best=100.0), 0.0)
        self.assertEqual(expected_improvement(50.0, 1e-15, f_best=100.0), 0.0)

    def test_increases_with_uncertainty(self) -> None:
        """At fixed mu, more uncertainty means more expected improvement."""
        values = [
            expected_improvement(mu=100.0, sigma=s, f_best=100.0)
            for s in (0.5, 1.0, 2.0, 8.0)
        ]
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[0])

    def test_at_incumbent_equals_half_sigma_scaled(self) -> None:
        """When mu == f_best, EI reduces to sigma * phi(0)."""
        sigma = 3.0
        expected = sigma * standard_normal_pdf(0.0)
        self.assertAlmostEqual(
            expected_improvement(mu=100.0, sigma=sigma, f_best=100.0),
            expected,
            places=12,
        )

    def test_xi_reduces_ei(self) -> None:
        base = expected_improvement(mu=90.0, sigma=2.0, f_best=100.0, xi=0.0)
        with_margin = expected_improvement(mu=90.0, sigma=2.0, f_best=100.0, xi=5.0)
        self.assertLess(with_margin, base)

    def test_dominant_term_for_large_improvement(self) -> None:
        """With a big, certain improvement, EI approaches the improvement."""
        ei = expected_improvement(mu=10.0, sigma=0.01, f_best=100.0)
        self.assertAlmostEqual(ei, 90.0, places=6)


class TestArgmax(unittest.TestCase):
    def setUp(self) -> None:
        # A tiny 1-D problem: candidate value IS its own feature.
        self.candidates = [1, 2, 3, 4, 5]
        self.encode = lambda c: [float(c)]

    def test_selects_highest_ei(self) -> None:
        # Predicted energy decreases with the candidate; uncertainty is equal.
        predict = lambda x: (100.0 - 10.0 * x[0], 2.0)  # noqa: E731
        best, score, scores = argmax_expected_improvement(
            self.candidates, self.encode, predict, f_best=100.0
        )
        self.assertEqual(best, 5)
        self.assertEqual(len(scores), len(self.candidates))
        self.assertAlmostEqual(score, max(scores))

    def test_respects_exclusions(self) -> None:
        predict = lambda x: (100.0 - 10.0 * x[0], 2.0)  # noqa: E731
        best, _score, scores = argmax_expected_improvement(
            self.candidates, self.encode, predict, f_best=100.0, exclude=[5, 4]
        )
        self.assertEqual(best, 3)
        self.assertEqual(scores[4], -math.inf)
        self.assertEqual(scores[3], -math.inf)

    def test_returns_none_when_all_excluded(self) -> None:
        predict = lambda x: (1.0, 1.0)  # noqa: E731
        best, score, _ = argmax_expected_improvement(
            self.candidates, self.encode, predict, f_best=1.0, exclude=self.candidates
        )
        self.assertIsNone(best)
        self.assertEqual(score, 0.0)

    def test_explores_uncertain_region(self) -> None:
        """A high-uncertainty candidate can beat a slightly better mean."""

        def predict(x):
            # Candidate 1 is marginally better but fully known; candidate 5 is
            # slightly worse in expectation but highly uncertain.
            return (95.0, 0.0) if x[0] == 1 else (99.0, 20.0)

        best, _score, _ = argmax_expected_improvement(
            self.candidates, self.encode, predict, f_best=100.0
        )
        self.assertNotEqual(
            best, 1, "EI should explore an uncertain candidate over a known one"
        )


if __name__ == "__main__":
    unittest.main()
