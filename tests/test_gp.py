"""Gaussian Process surrogate: linear algebra, posterior, and calibration.

The sklearn cross-check is skipped automatically when sklearn is absent, so the
suite still runs on a standard-library-only machine.
"""

from __future__ import annotations

import math
import random
import unittest

from bopis.gp import (
    GaussianProcess,
    HyperParams,
    LinAlgError,
    cho_solve,
    cholesky,
    log_det_from_cholesky,
    nelder_mead,
    rbf,
    solve_lower,
)


def _matmul_lower_transpose(lower):
    n = len(lower)
    return [
        [sum(lower[i][k] * lower[j][k] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]


class TestLinearAlgebra(unittest.TestCase):
    def test_cholesky_reconstructs(self) -> None:
        a = [[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]]
        lower = cholesky(a)
        product = _matmul_lower_transpose(lower)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(product[i][j], a[i][j], places=12)

    def test_cholesky_is_lower_triangular(self) -> None:
        a = [[4.0, 2.0], [2.0, 3.0]]
        lower = cholesky(a)
        self.assertEqual(lower[0][1], 0.0)

    def test_cholesky_rejects_non_positive_definite(self) -> None:
        with self.assertRaises(LinAlgError):
            cholesky([[1.0, 2.0], [2.0, 1.0]])

    def test_solve_lower(self) -> None:
        lower = [[2.0, 0.0], [1.0, 3.0]]
        x = solve_lower(lower, [4.0, 11.0])
        self.assertAlmostEqual(x[0], 2.0, places=12)
        self.assertAlmostEqual(x[1], 3.0, places=12)

    def test_cho_solve_matches_direct_solution(self) -> None:
        a = [[4.0, 1.0], [1.0, 3.0]]
        b = [1.0, 2.0]
        lower = cholesky(a)
        x = cho_solve(lower, b)
        # Verify a @ x == b
        for i in range(2):
            self.assertAlmostEqual(
                sum(a[i][j] * x[j] for j in range(2)), b[i], places=12
            )

    def test_log_det(self) -> None:
        a = [[4.0, 0.0], [0.0, 9.0]]
        lower = cholesky(a)
        self.assertAlmostEqual(
            log_det_from_cholesky(lower), math.log(36.0), places=12
        )


class TestKernel(unittest.TestCase):
    def test_equals_signal_variance_at_zero_distance(self) -> None:
        self.assertAlmostEqual(rbf([1.0, 2.0], [1.0, 2.0], 3.0, [0.5]), 3.0, places=12)

    def test_decays_with_distance(self) -> None:
        values = [
            rbf([0.0], [d], 1.0, [1.0]) for d in (0.0, 0.5, 1.0, 2.0, 4.0)
        ]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_is_symmetric(self) -> None:
        a, b = [0.1, 0.7], [0.9, 0.2]
        self.assertAlmostEqual(
            rbf(a, b, 2.0, [0.3]), rbf(b, a, 2.0, [0.3]), places=14
        )

    def test_ard_uses_per_dimension_scales(self) -> None:
        # A long scale on dim 0 makes distance along dim 0 nearly irrelevant.
        loose = rbf([0.0, 0.0], [1.0, 0.0], 1.0, [100.0, 0.1])
        tight = rbf([0.0, 0.0], [0.0, 1.0], 1.0, [100.0, 0.1])
        self.assertGreater(loose, tight)

    def test_known_value(self) -> None:
        # k = sigma_f^2 * exp(-d^2 / (2 l^2)); d=1, l=1 -> exp(-0.5)
        self.assertAlmostEqual(
            rbf([0.0], [1.0], 1.0, [1.0]), math.exp(-0.5), places=12
        )


class TestNelderMead(unittest.TestCase):
    def test_finds_minimum_of_quadratic(self) -> None:
        objective = lambda v: (v[0] - 3.0) ** 2 + (v[1] + 1.0) ** 2  # noqa: E731
        best, value = nelder_mead(objective, [0.0, 0.0], max_iter=800, tol=1e-12)
        self.assertAlmostEqual(best[0], 3.0, places=3)
        self.assertAlmostEqual(best[1], -1.0, places=3)
        self.assertLess(value, 1e-6)

    def test_respects_bounds(self) -> None:
        objective = lambda v: (v[0] - 10.0) ** 2  # noqa: E731
        best, _ = nelder_mead(objective, [0.0], bounds=[(-1.0, 1.0)])
        self.assertLessEqual(best[0], 1.0 + 1e-9)


class TestGaussianProcess(unittest.TestCase):
    def setUp(self) -> None:
        rng = random.Random(4)
        self.x = [[rng.random(), rng.random()] for _ in range(14)]
        self.f = lambda p: math.sin(3.0 * p[0]) + 0.5 * p[1] ** 2
        self.y = [self.f(p) for p in self.x]

    def test_interpolates_training_points_with_negligible_noise(self) -> None:
        """As sigma_n^2 -> 0 the posterior must pass through the data."""
        gp = GaussianProcess(optimize_hyperparameters=False).fit(self.x, self.y)
        gp.hyper = HyperParams(1.0, [0.5], 1e-10)
        gp._factorize(gp.hyper)
        for point, target in zip(self.x, self.y):
            mean, std = gp.predict(point)
            self.assertAlmostEqual(mean, target, places=6)
            self.assertLess(std, 1e-4)

    def test_reverts_to_prior_mean_far_from_data(self) -> None:
        """Far away, the posterior mean returns to the (standardized) prior."""
        gp = GaussianProcess(optimize_hyperparameters=False).fit(self.x, self.y)
        gp.hyper = HyperParams(1.0, [0.1], 1e-8)
        gp._factorize(gp.hyper)
        mean, std = gp.predict([50.0, 50.0])
        prior_mean = sum(self.y) / len(self.y)
        self.assertAlmostEqual(mean, prior_mean, places=4)
        self.assertGreater(std, 0.0)

    def test_uncertainty_grows_away_from_data(self) -> None:
        gp = GaussianProcess(seed=1).fit(self.x, self.y)
        _m_near, near = gp.predict(self.x[0])
        _m_far, far = gp.predict([9.0, 9.0])
        self.assertGreater(far, near)

    def test_predictive_std_exceeds_latent_std(self) -> None:
        """Observation noise must widen the interval used for calibration."""
        gp = GaussianProcess(seed=2).fit(self.x, self.y)
        _mean, latent = gp.predict(self.x[3])
        predictive = gp.predictive_std(self.x[3])
        self.assertGreaterEqual(predictive, latent)

    def test_hyperparameter_fit_improves_likelihood(self) -> None:
        gp = GaussianProcess(seed=3).fit(self.x, self.y)
        fitted = gp.log_marginal_likelihood_
        default = gp._log_marginal_likelihood(HyperParams(1.0, [0.5], 1e-4))
        self.assertGreaterEqual(fitted, default)

    def test_variance_never_negative(self) -> None:
        gp = GaussianProcess(seed=5).fit(self.x, self.y)
        for point in self.x + [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]:
            _mean, std = gp.predict(point)
            self.assertGreaterEqual(std, 0.0)

    def test_rejects_mismatched_inputs(self) -> None:
        with self.assertRaises(ValueError):
            GaussianProcess().fit([[0.0]], [1.0, 2.0])

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            GaussianProcess().fit([], [])

    def test_predict_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            GaussianProcess().predict([0.0])

    def test_handles_constant_targets(self) -> None:
        """A degenerate zero-variance target must not blow up standardization."""
        gp = GaussianProcess(optimize_hyperparameters=False).fit(
            [[0.0], [1.0], [2.0]], [5.0, 5.0, 5.0]
        )
        mean, std = gp.predict([0.5])
        self.assertAlmostEqual(mean, 5.0, places=6)
        self.assertGreaterEqual(std, 0.0)

    def test_leave_one_out_returns_one_row_per_observation(self) -> None:
        gp = GaussianProcess(seed=7).fit(self.x, self.y)
        loo = gp.leave_one_out()
        self.assertEqual(len(loo), len(self.x))
        for actual, mean, sd in loo:
            self.assertTrue(math.isfinite(actual))
            self.assertTrue(math.isfinite(mean))
            self.assertGreaterEqual(sd, 0.0)

    def test_ard_fit_produces_one_scale_per_dimension(self) -> None:
        gp = GaussianProcess(ard=True, seed=8, n_restarts=2).fit(self.x, self.y)
        self.assertEqual(len(gp.hyper.length_scales), 2)


class TestAgainstSklearn(unittest.TestCase):
    """Dev-only cross-check. Skipped when sklearn is not installed."""

    def setUp(self) -> None:
        try:
            import numpy  # noqa: F401
            import sklearn.gaussian_process  # noqa: F401
        except ImportError:
            self.skipTest("sklearn/numpy not installed (core does not need them)")

    def test_posterior_and_likelihood_match(self) -> None:
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF,
            ConstantKernel,
            WhiteKernel,
        )

        rng = random.Random(11)
        x = [[rng.random(), rng.random(), rng.random()] for _ in range(16)]
        y = [math.cos(2 * p[0]) + p[1] - 0.3 * p[2] for p in x]

        mine = GaussianProcess(seed=13).fit(x, y)
        hyper = mine.hyper

        kernel = (
            ConstantKernel(hyper.signal_variance, "fixed")
            * RBF(hyper.length_scales[0], "fixed")
            + WhiteKernel(hyper.noise_variance, "fixed")
        )
        mean_y = sum(y) / len(y)
        std_y = math.sqrt(sum((v - mean_y) ** 2 for v in y) / (len(y) - 1))
        theirs = GaussianProcessRegressor(
            kernel=kernel, optimizer=None, normalize_y=False
        ).fit(np.array(x), (np.array(y) - mean_y) / std_y)

        # Log marginal likelihood, on identical standardized targets.
        self.assertAlmostEqual(
            mine.log_marginal_likelihood_,
            float(theirs.log_marginal_likelihood_value_),
            places=8,
        )

        # Posterior mean at fresh points, un-standardized. Agreement is limited
        # only by floating-point accumulation: this is a pure-Python Cholesky
        # and triangular solve against LAPACK, so ~1e-7 absolute is the floor.
        probes = [[rng.random(), rng.random(), rng.random()] for _ in range(8)]
        their_mean, their_std = theirs.predict(np.array(probes), return_std=True)
        for i, probe in enumerate(probes):
            mean, _std = mine.predict(probe)
            self.assertAlmostEqual(mean, their_mean[i] * std_y + mean_y, places=6)

        # Standard deviation: sklearn's `return_std` evaluates the *full* kernel
        # diagonal, which includes the WhiteKernel term, so it reports the
        # standard deviation of a new **observation**. That corresponds to
        # `predictive_std` here, not to the latent-function std returned by
        # `predict`. Asserting both directions pins the semantics down: the
        # latent std must be strictly smaller, and the predictive std must match
        # sklearn.
        for i, probe in enumerate(probes):
            _mean, latent = mine.predict(probe)
            predictive = mine.predictive_std(probe)
            self.assertAlmostEqual(predictive, their_std[i] * std_y, places=7)
            self.assertLess(latent, predictive)


if __name__ == "__main__":
    unittest.main()
