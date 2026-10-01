"""Daily clean-market diagnostics; no historical price forcing or release labels.

One symbol per reference panel. Thirty returns require 31 closes; real windows
use disjoint 31-session chunks, while simulation includes its initial price.
All lag pairs stay inside an episode. No SciPy/pandas runtime dependency.
"""
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np


CALIBRATION_METRICS = ("volatility", "abs_acf1", "volume_cv", "volume_abs_return_corr", "max_drawdown")
VALIDATION_METRICS = ("return_q01", "excess_kurtosis", "abs_acf5", "largest_move", "drawdown_duration")


@dataclass(frozen=True)
class MarketWindow:
    identity: str
    returns: tuple[float, ...]  # simple split-adjusted price returns, not total returns
    volumes: tuple[float, ...]

    def __post_init__(self):
        if not self.identity or len(self.returns) != 30 or len(self.volumes) != 30:
            raise ValueError("a window requires identity and exactly 30 aligned daily observations")
        if not all(np.isfinite(r) and r > -1 for r in self.returns):
            raise ValueError("invalid simple return")
        if not all(np.isfinite(v) and v >= 0 for v in self.volumes):
            raise ValueError("invalid volume")


def source_fingerprint() -> str:
    """Bind executable source and configurations, including this diagnostic code."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.suffix in {".py", ".yaml"}):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def load_reference(path: Path, start: str, end: str, *, symbol: str) -> list[MarketWindow]:
    """Read Alpha Vantage DAILY_ADJUSTED JSON, not unverified CSV columns.

    Filter dates before inspecting values. Use declared split events to put OHLC
    and shares on a common basis; do not compare dividend-adjusted total returns
    against a simulator with no dividends. Final-test values are never used here.
    """
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("invalid reference period")
    payload = json.loads(path.read_text())
    if payload.get("Meta Data", {}).get("2. Symbol") != symbol:
        raise ValueError("reference symbol mismatch or provider error")
    selected = sorted((d, row) for d, row in payload["Time Series (Daily)"].items() if start <= d <= end)
    rows = []
    split_factor = 1.0
    previous = None
    for day, row in selected:
        current = date.fromisoformat(day)
        if previous and (current - previous).days > 7:
            raise ValueError("gap exceeds seven calendar days; inspect missing/suspended sessions")
        previous = current
        op, high, low, close = (float(row[f"{i}. {name}"]) for i, name in enumerate(("open", "high", "low", "close"), 1))
        volume, split = float(row["6. volume"]), float(row["8. split coefficient"])
        if not all(np.isfinite(x) for x in (op, high, low, close, volume, split)):
            raise ValueError("nonfinite OHLCV/split input")
        if not 0 < low <= min(op, close) <= max(op, close) <= high or volume < 0 or split <= 0:
            raise ValueError("inconsistent OHLCV/split input")
        split_factor *= split
        rows.append((day, close * split_factor, volume / split_factor))
    windows = []
    for offset in range(0, len(rows) - 30, 31):
        chunk = rows[offset:offset + 31]
        prices = np.asarray([r[1] for r in chunk])
        windows.append(MarketWindow(f"{symbol}:{chunk[0][0]}:{chunk[-1][0]}",
                                    tuple(prices[1:] / prices[:-1] - 1), tuple(r[2] for r in chunk[1:])))
    if not windows:
        raise ValueError("reference period contains no complete window")
    return windows


def correlation(left, right) -> float | None:
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize(windows: list[MarketWindow]) -> dict:
    if not windows or len({w.identity for w in windows}) != len(windows):
        raise ValueError("need unique independent window identities")
    returns = np.asarray([w.returns for w in windows])
    volumes = np.asarray([w.volumes for w in windows])
    features = {key: [] for key in ("volatility", "max_drawdown", "volume_cv", "volume_abs_return_corr", "largest_move", "drawdown_duration")}
    for r, v in zip(returns, volumes):
        prices = np.r_[1., np.cumprod(1 + r)]
        drawdown = 1 - prices / np.maximum.accumulate(prices)
        duration = longest = 0
        for value in drawdown[1:]:
            duration = duration + 1 if value > 1e-12 else 0
            longest = max(longest, duration)
        features["volatility"].append(float(np.std(r, ddof=1)))
        features["max_drawdown"].append(float(np.max(drawdown)))
        features["volume_cv"].append(float(np.std(v, ddof=1) / np.mean(v)) if np.mean(v) > 0 else None)
        features["volume_abs_return_corr"].append(correlation(v, np.abs(r)))
        features["largest_move"].append(float(np.max(np.abs(r))))
        features["drawdown_duration"].append(longest)
    metrics = {key: float(np.mean([x for x in values if x is not None])) if any(x is not None for x in values) else None
               for key, values in features.items()}
    pooled = returns.ravel()
    variance = float(np.var(pooled))
    metrics.update(return_q01=float(np.quantile(pooled, .01)),
                   excess_kurtosis=float(np.mean((pooled - np.mean(pooled)) ** 4) / variance ** 2 - 3) if variance > 1e-24 else None)
    for lag in (1, 5, 10):
        # Never concatenate independent episode ends with the next episode's start.
        metrics[f"abs_acf{lag}"] = correlation(np.abs(returns[:, :-lag]).ravel(), np.abs(returns[:, lag:]).ravel())
        metrics[f"return_acf{lag}"] = correlation(returns[:, :-lag].ravel(), returns[:, lag:].ravel())
    return {"windows": len(windows), "metrics": metrics, "window_features": features,
            "metric_coverage": {key: sum(x is not None for x in values) / len(values) for key, values in features.items()}}


def wasserstein_1(left, right) -> float:
    """Exact 1-D empirical W1, with arbitrary sample sizes and equal weights."""
    left, right = np.sort(left), np.sort(right)
    if not len(left) or not len(right) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("W1 requires nonempty finite samples")
    grid = np.unique(np.r_[left, right])
    gap = np.abs(np.searchsorted(left, grid[:-1], side="right") / len(left)
                 - np.searchsorted(right, grid[:-1], side="right") / len(right))
    return float(np.sum(gap * np.diff(grid)))


def compare_distributions(simulated, reference, *, seed=1701, repetitions=200) -> dict:
    """Exploratory equal-n real-real reference, not a calibrated acceptance test.

    Real groups are disjoint on each draw. Single-symbol windows retain regime
    dependence; random grouping is not a confidence proof under arbitrary shifts.
    Quantile intervals describe subsampling variability, not parameter uncertainty.
    """
    sim, real = summarize(simulated), summarize(reference)
    rng = np.random.default_rng(seed)
    n = min(len(simulated), len(reference) // 2)
    if n < 2 or repetitions < 2:
        raise ValueError("comparison needs at least two matched windows and repetitions")
    result = {}
    for key, sim_values in sim["window_features"].items():
        real_values = real["window_features"][key]
        if any(x is None for x in [*sim_values, *real_values]):
            result[key] = {"status": "undefined_metric", "matched_n": n}
            continue
        a, b = np.asarray(sim_values), np.asarray(real_values)
        rr, sr = [], []
        for _ in range(repetitions):
            indices = rng.permutation(len(b))[:2 * n]
            rr.append(wasserstein_1(b[indices[:n]], b[indices[n:]]))
            sr.append(wasserstein_1(rng.choice(a, n, replace=False), b[indices[:n]]))
        result[key] = {"matched_n": n, "full_sample_w1": wasserstein_1(a, b),
                       "matched_sim_real_median": float(np.median(sr)),
                       "matched_sim_real_interval": np.quantile(sr, [.025, .975]).tolist(),
                       "real_real_q95": float(np.quantile(rr, .95)),
                       "within_empirical_reference": bool(np.median(sr) <= np.quantile(rr, .95))}
    return result


def baseline_windows(calibration, *, kind: str, count: int, seed: int) -> list[MarketWindow]:
    """Two calibration-only references; no held-out returns used for fitting."""
    rng = np.random.default_rng(seed)
    returns = np.asarray([w.returns for w in calibration])
    volumes = np.asarray([w.volumes for w in calibration])
    logs = np.log1p(returns)
    result = []
    for i in range(count):
        if kind == "gaussian":
            r = np.expm1(rng.normal(np.mean(logs), np.std(logs), 30))
            # Explicit independence reference, not a fitted volume model.
            v = rng.choice(volumes.ravel(), 30)
        elif kind == "block_bootstrap":
            blocks = [(rng.integers(len(returns)), rng.integers(26)) for _ in range(6)]
            r = np.concatenate([returns[j, start:start + 5] for j, start in blocks])
            v = np.concatenate([volumes[j, start:start + 5] for j, start in blocks])
        else:
            raise ValueError("unknown baseline")
        result.append(MarketWindow(f"{kind}-{seed}-{i}", tuple(r), tuple(v)))
    return result
