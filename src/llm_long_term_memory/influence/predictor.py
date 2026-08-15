"""A cheap stand-in for the ablation, so the packer can run per query.

Ridge regression on the features in `features.py`, fitted by normal equations in
numpy. The model is small on purpose:

  * the label set is at most `questions x top_k` rows — hundreds, not millions
  * it must evaluate in microseconds inside the packer
  * a linear model's coefficients are readable, and "which signal did it learn to
    trust" is the question the experiment is asking

**Splitting is by question, never by row.** Twenty memories retrieved for one
question share its query terms, its namespace, and its answer; putting some in
train and the rest in test leaks, and the resulting score would report memorisation
of the dev questions as predictive power.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES


@dataclass(slots=True)
class FitReport:
    n_train: int
    n_test: int
    rmse: float
    baseline_rmse: float
    """RMSE of predicting the training mean. A model that cannot beat this has
    learned nothing."""
    spearman_vs_relevance: float
    """Rank correlation between predicted utility and the retrieval score it was
    given as a feature. Near 1.0 means the predictor is relevance in disguise and
    the whole exercise is answered in the negative."""
    coefficients: dict[str, float]

    @property
    def beats_baseline(self) -> bool:
        return self.rmse < self.baseline_rmse


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / denom) if denom else 0.0


class UtilityPredictor:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def _standardise(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean = x.mean(axis=0)
            # Guard constant columns: an all-zero feature would divide by zero and
            # poison every prediction with NaN.
            self.scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
        return (x - self.mean) / self.scale

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        z = self._standardise(np.asarray(x, dtype=float), fit=True)
        z = np.hstack([z, np.ones((len(z), 1))])  # bias
        penalty = self.alpha * np.eye(z.shape[1])
        penalty[-1, -1] = 0.0  # never shrink the intercept
        self.weights = np.linalg.solve(z.T @ z + penalty, z.T @ np.asarray(y, dtype=float))

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("fit() first")
        z = self._standardise(np.asarray(x, dtype=float))
        return np.hstack([z, np.ones((len(z), 1))]) @ self.weights

    @property
    def coefficients(self) -> dict[str, float]:
        if self.weights is None:
            return {}
        return dict(zip(FEATURE_NAMES, self.weights[:-1].tolist(), strict=False))

    def save(self, path: str | Path) -> None:
        if self.weights is None:
            raise RuntimeError("fit() before save(); there is nothing to persist")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "weights": self.weights.tolist(),
                    "mean": self.mean.tolist(),
                    "scale": self.scale.tolist(),
                    "features": list(FEATURE_NAMES),
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> UtilityPredictor:
        """Refuses a model whose feature layout no longer matches the code.

        Weights are positional. A feature added, removed, or reordered since the
        model was fitted produces no shape error and no exception — every utility
        is simply computed against the wrong columns, and because the packer only
        consumes the ranking the sole symptom is quietly worse selection.
        """
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        saved = tuple(d.get("features", ()))
        if saved and saved != FEATURE_NAMES:
            raise ValueError(
                f"model was fitted on {len(saved)} features {saved!r}, but the code now "
                f"defines {len(FEATURE_NAMES)} {FEATURE_NAMES!r}. Refit rather than "
                "predicting against mismatched columns."
            )
        m = cls(alpha=d["alpha"])
        m.weights = np.array(d["weights"])
        m.mean = np.array(d["mean"])
        m.scale = np.array(d["scale"])
        return m


RELEVANCE_COLUMN = FEATURE_NAMES.index("retrieval_score")


def fit_grouped(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    folds: int = 5,
    alpha: float = 1.0,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> tuple[UtilityPredictor, FitReport]:
    """Cross-validate with questions held out whole, then refit on everything.

    `feature_names` is accepted so the caller's column order can be checked rather
    than assumed. `spearman_vs_relevance` decides whether the experiment answered
    its question at all, and reading the wrong column would make that verdict
    confidently wrong with nothing to notice.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError(f"feature order {feature_names!r} does not match {FEATURE_NAMES!r}")
    if x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} columns, got {x.shape[1]}")
    unique = sorted(set(groups))
    assignment = {g: i % folds for i, g in enumerate(unique)}
    fold_of = np.array([assignment[g] for g in groups])

    preds, truth, relevance = [], [], []
    for k in range(folds):
        train, test = fold_of != k, fold_of == k
        if train.sum() < 2 or test.sum() == 0:
            continue
        model = UtilityPredictor(alpha=alpha)
        model.fit(x[train], y[train])
        preds.append(model.predict(x[test]))
        truth.append(y[test])
        relevance.append(x[test][:, RELEVANCE_COLUMN])

    if preds:
        p, t, r = np.concatenate(preds), np.concatenate(truth), np.concatenate(relevance)
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        baseline = float(np.sqrt(((t.mean() - t) ** 2).mean()))
        corr = _spearman(p, r)
        n_test = len(t)
    else:
        rmse = baseline = corr = 0.0
        n_test = 0

    final = UtilityPredictor(alpha=alpha)
    final.fit(x, y)
    return final, FitReport(
        n_train=len(y),
        n_test=n_test,
        rmse=rmse,
        baseline_rmse=baseline,
        spearman_vs_relevance=corr,
        coefficients=final.coefficients,
    )
