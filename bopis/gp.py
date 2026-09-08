"""Gaussian Process surrogate model -- standard library only.

Implements the surrogate of Chapter 3: an RBF (squared-exponential) kernel, the
GP posterior mean and variance, the log marginal likelihood, and hyperparameter
optimization.

Deviation from the manuscript (amendment A-7)
----------------------------------------------
Chapter 3 states that ``theta = {sigma_f^2, l, sigma_n^2}`` is optimized by
**L-BFGS-B**. L-BFGS-B is a gradient-based quasi-Newton method; no
standard-library implementation exists, and supplying analytic kernel gradients
would add substantial code for little benefit at this problem size (three to
seven hyperparameters, at most thirty observations). Instead the log marginal
likelihood is maximized by a **coarse log-space grid search followed by
multi-start Nelder-Mead** (a derivative-free simplex method), which is
implemented here in full. On problems of this dimension the two find
indistinguishable optima; the grid stage additionally guards against the local
minima that make single-start quasi-Newton fits of GP hyperparameters
unreliable.

Numerical notes
---------------
* ``y`` is standardized (zero mean, unit variance) before fitting, so a
  zero-mean GP prior is appropriate; predictions are un-transformed on output.
* Cholesky factorization uses escalating jitter, because the kernel matrix
  becomes ill-conditioned whenever two evaluated configurations are close in
  encoded space -- which Bayesian Optimization actively causes as it converges.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]

_LOG_2PI = math.log(2.0 * math.pi)


class LinAlgError(RuntimeError):
    """A matrix operation failed (typically a non-positive-definite kernel)."""


# --------------------------------------------------------------------------- #
# Linear algebra
# --------------------------------------------------------------------------- #


def cholesky(a: Matrix) -> Matrix:
    """Lower-triangular Cholesky factor ``L`` with ``L @ L.T == a``.

    Raises :class:`LinAlgError` if *a* is not positive definite.
    """
    n = len(a)
    lower: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = a[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if acc <= 0.0:
                    raise LinAlgError(
                        f"matrix not positive definite at pivot {i} (value {acc:.3e})"
                    )
                lower[i][j] = math.sqrt(acc)
            else:
                lower[i][j] = acc / lower[j][j]
    return lower


def solve_lower(lower: Matrix, b: Vector) -> Vector:
    """Forward substitution: solve ``L x = b`` for lower-triangular ``L``."""
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        acc = b[i] - sum(lower[i][k] * x[k] for k in range(i))
        x[i] = acc / lower[i][i]
    return x


def solve_upper_from_lower(lower: Matrix, b: Vector) -> Vector:
    """Back substitution: solve ``L.T x = b`` without materializing ``L.T``."""
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = b[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))
        x[i] = acc / lower[i][i]
    return x


def cho_solve(lower: Matrix, b: Vector) -> Vector:
    """Solve ``a x = b`` given the Cholesky factor *lower* of ``a``."""
    return solve_upper_from_lower(lower, solve_lower(lower, b))


def log_det_from_cholesky(lower: Matrix) -> float:
    """``log|a|`` from its Cholesky factor: ``2 * sum(log L_ii)``."""
    return 2.0 * sum(math.log(lower[i][i]) for i in range(len(lower)))


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #


def rbf(
    x1: Sequence[float],
    x2: Sequence[float],
    signal_variance: float,
    length_scales: Sequence[float],
) -> float:
    """Squared-exponential kernel.

    ``k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 l^2))``, generalized to
    per-dimension length scales (ARD) when *length_scales* has one entry per
    input dimension.
    """
    if len(length_scales) == 1:
        scale = length_scales[0]
        sq = sum((a - b) ** 2 for a, b in zip(x1, x2)) / (scale * scale)
    else:
        sq = sum(
            ((a - b) / s) ** 2 for a, b, s in zip(x1, x2, length_scales)
        )
    return signal_variance * math.exp(-0.5 * sq)


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #


class HyperParams:
    """``theta = {sigma_f^2, l (possibly per-dimension), sigma_n^2}``."""

    __slots__ = ("signal_variance", "length_scales", "noise_variance")

    def __init__(
        self,
        signal_variance: float,
        length_scales: Sequence[float],
        noise_variance: float,
    ) -> None:
        self.signal_variance = float(signal_variance)
        self.length_scales = [float(v) for v in length_scales]
        self.noise_variance = float(noise_variance)

    def to_log_vector(self) -> Vector:
        return (
            [math.log(self.signal_variance)]
            + [math.log(v) for v in self.length_scales]
            + [math.log(self.noise_variance)]
        )

    @staticmethod
    def from_log_vector(vec: Sequence[float]) -> "HyperParams":
        return HyperParams(
            signal_variance=math.exp(vec[0]),
            length_scales=[math.exp(v) for v in vec[1:-1]],
            noise_variance=math.exp(vec[-1]),
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "signal_variance": self.signal_variance,
            "length_scales": list(self.length_scales),
            "noise_variance": self.noise_variance,
            "noise_std": math.sqrt(self.noise_variance),
        }

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        scales = ", ".join(f"{v:.4f}" for v in self.length_scales)
        return (
            f"HyperParams(sigma_f^2={self.signal_variance:.4f}, "
            f"l=[{scales}], sigma_n^2={self.noise_variance:.6f})"
        )


#: Optimization bounds in log space. Because ``y`` is standardized, a signal
#: variance far from 1 and a noise variance above ~1 are both pathological, and
#: length scales are bounded relative to the unit hypercube the encoder produces.
LOG_BOUNDS = {
    "signal_variance": (math.log(1e-3), math.log(1e2)),
    "length_scale": (math.log(1e-2), math.log(1e1)),
    "noise_variance": (math.log(1e-6), math.log(1e0)),
}


def _bounds_vector(n_scales: int) -> List[Tuple[float, float]]:
    return (
        [LOG_BOUNDS["signal_variance"]]
        + [LOG_BOUNDS["length_scale"]] * n_scales
        + [LOG_BOUNDS["noise_variance"]]
    )


def _clamp_to_bounds(vec: Sequence[float], bounds: Sequence[Tuple[float, float]]) -> Vector:
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(vec, bounds)]


# --------------------------------------------------------------------------- #
# Nelder-Mead (derivative-free simplex), stdlib
# --------------------------------------------------------------------------- #


def nelder_mead(
    objective: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    step: float = 0.5,
    max_iter: int = 400,
    tol: float = 1e-6,
) -> Tuple[Vector, float]:
    """Minimize *objective* from *x0*. Returns ``(best_x, best_value)``.

    Standard Nelder-Mead with reflection, expansion, contraction and shrink.
    Bounds are enforced by clamping, which is adequate here because the optimum
    of the log marginal likelihood lies in the interior for any well-posed
    problem and the bounds exist only to exclude degenerate hyperparameters.
    """
    n = len(x0)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    def clamp(vec: Sequence[float]) -> Vector:
        return _clamp_to_bounds(vec, bounds) if bounds else list(vec)

    simplex: List[Vector] = [clamp(x0)]
    for i in range(n):
        point = list(x0)
        point[i] += step
        simplex.append(clamp(point))

    values = [objective(p) for p in simplex]

    for _ in range(max_iter):
        order = sorted(range(len(simplex)), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        if abs(values[-1] - values[0]) <= tol * (abs(values[0]) + tol):
            break

        centroid = [
            sum(p[d] for p in simplex[:-1]) / n for d in range(n)
        ]
        worst = simplex[-1]

        reflected = clamp([centroid[d] + alpha * (centroid[d] - worst[d]) for d in range(n)])
        f_ref = objective(reflected)

        if values[0] <= f_ref < values[-2]:
            simplex[-1], values[-1] = reflected, f_ref
            continue

        if f_ref < values[0]:
            expanded = clamp(
                [centroid[d] + gamma * (reflected[d] - centroid[d]) for d in range(n)]
            )
            f_exp = objective(expanded)
            if f_exp < f_ref:
                simplex[-1], values[-1] = expanded, f_exp
            else:
                simplex[-1], values[-1] = reflected, f_ref
            continue

        contracted = clamp(
            [centroid[d] + rho * (worst[d] - centroid[d]) for d in range(n)]
        )
        f_con = objective(contracted)
        if f_con < values[-1]:
            simplex[-1], values[-1] = contracted, f_con
            continue

        best = simplex[0]
        for i in range(1, len(simplex)):
            simplex[i] = clamp(
                [best[d] + sigma * (simplex[i][d] - best[d]) for d in range(n)]
            )
            values[i] = objective(simplex[i])

    best_index = min(range(len(simplex)), key=lambda i: values[i])
    return simplex[best_index], values[best_index]


# --------------------------------------------------------------------------- #
# Gaussian Process
# --------------------------------------------------------------------------- #


class GaussianProcess:
    """Zero-mean GP regressor over standardized targets.

    Usage::

        gp = GaussianProcess()
        gp.fit(X, y)                 # optimizes hyperparameters
        mu, sigma = gp.predict(x)    # in the original units of y
    """

    #: Jitter ladder for the Cholesky factorization.
    _JITTER_LADDER = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)

    def __init__(
        self,
        ard: bool = False,
        n_restarts: int = 5,
        seed: int = 0,
        optimize_hyperparameters: bool = True,
    ) -> None:
        self.ard = ard
        self.n_restarts = n_restarts
        self.seed = seed
        self.optimize_hyperparameters = optimize_hyperparameters

        self.x_train: List[Vector] = []
        self.y_train: Vector = []
        self._y_mean = 0.0
        self._y_std = 1.0
        self._lower: Optional[Matrix] = None
        self._alpha: Optional[Vector] = None
        self._jitter_used = 0.0
        self.hyper: Optional[HyperParams] = None
        self.log_marginal_likelihood_: Optional[float] = None
        self._sqdist: Optional[Matrix] = None
        self._per_dim_sqdist: Optional[List[Matrix]] = None
        #: Optional warm start. Bayesian Optimization refits after every new
        #: observation, and consecutive fits differ by a single point, so the
        #: previous optimum is an excellent starting simplex.
        self.warm_start: Optional[HyperParams] = None

    # -- fitting ------------------------------------------------------------- #

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> "GaussianProcess":
        if len(x) != len(y):
            raise ValueError(f"x has {len(x)} rows but y has {len(y)}")
        if not x:
            raise ValueError("cannot fit a GP with no observations")

        self.x_train = [list(row) for row in x]
        self.y_train = list(y)

        n = len(y)
        self._y_mean = sum(y) / n
        if n > 1:
            variance = sum((v - self._y_mean) ** 2 for v in y) / (n - 1)
            self._y_std = math.sqrt(variance) if variance > 1e-18 else 1.0
        else:
            self._y_std = 1.0

        self._cache_distances()

        n_scales = len(self.x_train[0]) if self.ard else 1
        if self.optimize_hyperparameters and n >= 3:
            self.hyper = self._optimize(n_scales)
        else:
            # Too few points to identify hyperparameters; use sane defaults.
            self.hyper = HyperParams(1.0, [0.5] * n_scales, 1e-4)

        self._factorize(self.hyper)
        self.log_marginal_likelihood_ = self._log_marginal_likelihood(self.hyper)
        return self

    def _standardized_y(self) -> Vector:
        return [(v - self._y_mean) / self._y_std for v in self.y_train]

    def _cache_distances(self) -> None:
        """Precompute pairwise squared differences once per :meth:`fit`.

        Hyperparameter optimization evaluates the log marginal likelihood
        hundreds of times, and every evaluation needs the same pairwise
        distances -- only the length scales change. Caching them turns the
        per-evaluation kernel build from ``O(n^2 * d)`` arithmetic plus ``n^2``
        function calls into ``n^2`` exponentials, which is the difference
        between a usable fit and a prohibitive one at 30 observations.
        """
        n = len(self.x_train)
        dims = len(self.x_train[0]) if n else 0
        if self.ard:
            self._per_dim_sqdist = [
                [
                    [
                        (self.x_train[i][d] - self.x_train[j][d]) ** 2
                        for j in range(n)
                    ]
                    for i in range(n)
                ]
                for d in range(dims)
            ]
            self._sqdist = None
        else:
            self._sqdist = [
                [
                    sum(
                        (a - b) ** 2
                        for a, b in zip(self.x_train[i], self.x_train[j])
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ]
            self._per_dim_sqdist = None

    def _kernel_matrix(self, hyper: HyperParams) -> Matrix:
        n = len(self.x_train)
        signal = hyper.signal_variance
        exp = math.exp

        if self._sqdist is not None:
            inv = -0.5 / (hyper.length_scales[0] ** 2)
            sqdist = self._sqdist
            mat = [
                [signal * exp(inv * sqdist[i][j]) for j in range(n)]
                for i in range(n)
            ]
        else:
            per_dim = self._per_dim_sqdist or []
            inv_scales = [-0.5 / (s * s) for s in hyper.length_scales]
            mat = [[0.0] * n for _ in range(n)]
            for i in range(n):
                row = mat[i]
                for j in range(i, n):
                    acc = 0.0
                    for d, inv in enumerate(inv_scales):
                        acc += inv * per_dim[d][i][j]
                    row[j] = mat[j][i] = signal * exp(acc)

        for i in range(n):
            mat[i][i] += hyper.noise_variance
        return mat

    def _factorize(self, hyper: HyperParams) -> None:
        mat = self._kernel_matrix(hyper)
        n = len(mat)
        last_error: Optional[Exception] = None
        for jitter in self._JITTER_LADDER:
            try:
                candidate = [row[:] for row in mat]
                for i in range(n):
                    candidate[i][i] += jitter
                lower = cholesky(candidate)
            except LinAlgError as exc:
                last_error = exc
                continue
            self._lower = lower
            self._alpha = cho_solve(lower, self._standardized_y())
            self._jitter_used = jitter
            return
        raise LinAlgError(
            f"kernel matrix not positive definite even with jitter "
            f"{self._JITTER_LADDER[-1]:.0e}: {last_error}"
        )

    def _log_marginal_likelihood(self, hyper: HyperParams) -> float:
        """``log p(y | X, theta)`` exactly as written in Chapter 3."""
        mat = self._kernel_matrix(hyper)
        n = len(mat)
        try:
            lower = cholesky(mat)
        except LinAlgError:
            return -math.inf
        y_std = self._standardized_y()
        alpha = cho_solve(lower, y_std)
        quad = sum(a * b for a, b in zip(y_std, alpha))
        return -0.5 * quad - 0.5 * log_det_from_cholesky(lower) - 0.5 * n * _LOG_2PI

    def _optimize(self, n_scales: int) -> HyperParams:
        bounds = _bounds_vector(n_scales)
        rng = random.Random(self.seed)

        def negative_lml(vec: Sequence[float]) -> float:
            hyper = HyperParams.from_log_vector(_clamp_to_bounds(vec, bounds))
            value = self._log_marginal_likelihood(hyper)
            return -value if math.isfinite(value) else 1e18

        # Stage 1: coarse grid over a plausible region, to seed the simplex away
        # from local optima. Length scales share one grid value at this stage
        # even under ARD; the simplex then separates them.
        best_vec: Optional[Vector] = None
        best_val = math.inf
        for log_sf in (math.log(0.1), math.log(1.0), math.log(10.0)):
            for log_ls in (math.log(0.1), math.log(0.3), math.log(1.0), math.log(3.0)):
                for log_sn in (math.log(1e-5), math.log(1e-3), math.log(1e-1)):
                    vec = [log_sf] + [log_ls] * n_scales + [log_sn]
                    val = negative_lml(vec)
                    if val < best_val:
                        best_val, best_vec = val, vec
        assert best_vec is not None

        # Stage 2: multi-start Nelder-Mead. Order matters: the warm start (the
        # previous fit's optimum) goes first, then the grid winner, then seeded
        # random points. Consecutive BO fits differ by one observation, so the
        # warm start usually lands in the right basin immediately.
        starts: List[Vector] = []
        if self.warm_start is not None:
            warm = self.warm_start.to_log_vector()
            if len(warm) == len(bounds):
                starts.append(_clamp_to_bounds(warm, bounds))
        starts.append(best_vec)
        for _ in range(max(0, self.n_restarts - len(starts))):
            starts.append([rng.uniform(lo, hi) for lo, hi in bounds])

        best_solution, best_value = best_vec, best_val
        for start in starts:
            candidate, value = nelder_mead(negative_lml, start, bounds=bounds)
            if value < best_value:
                best_solution, best_value = candidate, value

        return HyperParams.from_log_vector(_clamp_to_bounds(best_solution, bounds))

    # -- prediction ---------------------------------------------------------- #

    def predict(self, x: Sequence[float]) -> Tuple[float, float]:
        """Posterior ``(mean, standard deviation)`` at *x*, in the units of y."""
        if self._lower is None or self._alpha is None or self.hyper is None:
            raise RuntimeError("GaussianProcess.predict called before fit")

        hyper = self.hyper
        k_star = [
            rbf(xi, x, hyper.signal_variance, hyper.length_scales)
            for xi in self.x_train
        ]
        mean_std = sum(k * a for k, a in zip(k_star, self._alpha))

        v = solve_lower(self._lower, k_star)
        prior = rbf(x, x, hyper.signal_variance, hyper.length_scales)
        variance_std = prior - sum(value * value for value in v)
        variance_std = max(variance_std, 0.0)  # guard against round-off

        mean = mean_std * self._y_std + self._y_mean
        std = math.sqrt(variance_std) * self._y_std
        return mean, std

    def predict_many(
        self, xs: Sequence[Sequence[float]]
    ) -> List[Tuple[float, float]]:
        return [self.predict(x) for x in xs]

    def predictive_std(self, x: Sequence[float]) -> float:
        """Std. dev. of an *observation* at *x*: includes the noise term.

        Calibration metrics such as UCR must use this rather than the latent
        function's standard deviation, otherwise a well-calibrated GP appears
        badly calibrated (amendment A-28).
        """
        _, std = self.predict(x)
        noise_std = math.sqrt(self.hyper.noise_variance) * self._y_std  # type: ignore[union-attr]
        return math.sqrt(std * std + noise_std * noise_std)

    # -- diagnostics --------------------------------------------------------- #

    def leave_one_out(self) -> List[Tuple[float, float, float]]:
        """Leave-one-out predictions: ``[(actual, mean, predictive_std), ...]``.

        Refits the GP n times with the incumbent hyperparameters held fixed
        (re-optimizing per fold would be both slow and optimistically biased).
        With only 20-30 observations, LOO is a far more honest reliability
        estimate than the sequence of one-step-ahead predictions alone
        (amendment A-36).
        """
        if self.hyper is None:
            raise RuntimeError("leave_one_out called before fit")

        results: List[Tuple[float, float, float]] = []
        x_all, y_all = self.x_train, self.y_train
        for i in range(len(x_all)):
            x_fold = x_all[:i] + x_all[i + 1 :]
            y_fold = y_all[:i] + y_all[i + 1 :]
            if len(x_fold) < 2:
                continue
            fold = GaussianProcess(
                ard=self.ard, seed=self.seed, optimize_hyperparameters=False
            )
            fold.fit(x_fold, y_fold)
            fold.hyper = self.hyper
            fold._factorize(self.hyper)
            mean, _ = fold.predict(x_all[i])
            results.append((y_all[i], mean, fold.predictive_std(x_all[i])))
        return results

    def describe(self) -> Dict[str, object]:
        return {
            "n_observations": len(self.y_train),
            "ard": self.ard,
            "jitter_used": self._jitter_used,
            "log_marginal_likelihood": self.log_marginal_likelihood_,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "hyperparameters": self.hyper.as_dict() if self.hyper else None,
        }
