"""Dependency-light benchmark shortcut baselines with trajectory folds."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class _Node:
    prediction: int
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None


class _CART:
    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self.root: _Node | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_CART":
        self.root = self._build(x, y, 0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("tree is not fitted")
        output = []
        for row in x:
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            output.append(node.prediction)
        return np.asarray(output, dtype=int)

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        prediction = Counter(map(int, y)).most_common(1)[0][0]
        node = _Node(prediction)
        if depth >= self.max_depth or len(set(map(int, y))) <= 1:
            return node
        best = None
        for feature in range(x.shape[1]):
            unique = np.unique(x[:, feature])
            thresholds = (unique[:-1] + unique[1:]) / 2.0
            for threshold in thresholds:
                left = x[:, feature] <= threshold
                if not left.any() or left.all():
                    continue
                score = left.mean() * _gini(y[left]) + (~left).mean() * _gini(y[~left])
                candidate = (score, feature, float(threshold), left)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None:
            return node
        _, node.feature, node.threshold, left = best
        node.left = self._build(x[left], y[left], depth + 1)
        node.right = self._build(x[~left], y[~left], depth + 1)
        return node


class _Softmax:
    def fit(self, x: np.ndarray, y: np.ndarray, classes: int) -> "_Softmax":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        self.weights = np.zeros((z.shape[1], classes), dtype=float)
        target = np.eye(classes)[y]
        for _ in range(800):
            logits = z @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            probability = np.exp(logits)
            probability /= probability.sum(axis=1, keepdims=True)
            gradient = z.T @ (probability - target) / len(z)
            gradient[1:] += 1e-3 * self.weights[1:]
            self.weights -= 0.15 * gradient
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        return np.argmax(z @ self.weights, axis=1)


class _RandomForest:
    """Small deterministic bagged random-subspace forest for audit use."""

    def __init__(self, trees: int = 40, max_depth: int = 5, seed: int = 2718) -> None:
        self.trees = trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_RandomForest":
        rng = np.random.default_rng(self.seed)
        feature_count = max(1, int(np.sqrt(x.shape[1])))
        self.estimators = []
        for _ in range(self.trees):
            rows = rng.integers(0, len(x), size=len(x))
            features = np.sort(rng.choice(x.shape[1], size=feature_count, replace=False))
            tree = _CART(self.max_depth).fit(x[rows][:, features], y[rows])
            self.estimators.append((features, tree))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        votes = np.stack([
            tree.predict(x[:, features]) for features, tree in self.estimators
        ])
        return np.asarray([
            Counter(map(int, votes[:, index])).most_common(1)[0][0]
            for index in range(votes.shape[1])
        ], dtype=int)


def build_shortcut_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) < 20:
        raise ValueError("shortcut audit requires a completed pilot")
    feature_names = [
        "day", "society_size",
        *[f"market.{name}" for name in sorted(records[0]["observation"]["market"])],
        *[f"social.{name}" for name in sorted(records[0]["observation"]["social"])],
        *[
            f"assets.{index}.{name}"
            for index, asset in enumerate(records[0]["observation"]["assets"])
            for name in sorted(key for key in asset if key != "asset_id")
        ],
        *[
            f"entities.{index}.{name}"
            for index, entity in enumerate(records[0]["observation"]["entities"])
            for name in sorted(key for key in entity if key != "entity_id")
        ],
    ]
    x = np.asarray([_public_features(row, feature_names) for row in records], dtype=float)
    trusted_x = np.asarray([_trusted_features(row) for row in records], dtype=float)
    folds = np.asarray([_fold(row["trajectory_id"]) for row in records])
    tasks = {
        "RF": [row["risk_regime"] for row in records],
        "MD": [row["mechanism"] for row in records],
        "IS": [row["oracle"]["primary_action"]["type"] for row in records],
    }
    task_reports = {}
    for task, labels in tasks.items():
        names = sorted(set(labels))
        encoded = np.asarray([names.index(label) for label in labels], dtype=int)
        single = {
            feature_names[index]: _cross_validated_accuracy(
                x[:, [index]], encoded, folds, "tree", 3
            )
            for index in range(x.shape[1])
        }
        results = {
            "uniform_random_expected_accuracy": 1.0 / len(names),
            "majority_accuracy": max(Counter(labels).values()) / len(labels),
            "day_only_tree_depth3": single["day"],
            "society_size_only_tree_depth3": single["society_size"],
            "best_single_feature_tree_depth3": {
                "feature": max(single, key=single.get),
                "accuracy": max(single.values()),
            },
            "aggregate_logistic": _cross_validated_accuracy(
                x, encoded, folds, "logistic", len(names)
            ),
            "aggregate_tree_depth2": _cross_validated_accuracy(
                x, encoded, folds, "tree", 2
            ),
            "aggregate_tree_depth3": _cross_validated_accuracy(
                x, encoded, folds, "tree", 3
            ),
            "public_random_forest": _cross_validated_accuracy(
                x, encoded, folds, "forest", len(names)
            ),
            "trusted_alpha_k_control_tree_depth3": _cross_validated_accuracy(
                trusted_x, encoded, folds, "tree", 3
            ),
        }
        results["shortcut_risk_above_90_percent"] = any(
            float(results[key]) > 0.90
            for key in (
                "day_only_tree_depth3", "society_size_only_tree_depth3",
                "aggregate_logistic", "aggregate_tree_depth2", "aggregate_tree_depth3",
                "public_random_forest",
            )
        ) or results["best_single_feature_tree_depth3"]["accuracy"] > 0.90
        task_reports[task] = results
    return {
        "schema_version": "wolfgym-shortcut-audit-v2",
        "record_count": len(records),
        "split_unit": "trajectory_id",
        "folds": 5,
        "public_feature_names": feature_names,
        "trusted_alpha_k_control_is_not_model_input": True,
        "tasks": task_reports,
        "gate": {
            "RF_no_public_shortcut_above_90_percent": not task_reports["RF"][
                "shortcut_risk_above_90_percent"
            ],
            "IS_no_public_shortcut_above_90_percent": not task_reports["IS"][
                "shortcut_risk_above_90_percent"
            ],
            "MD_no_public_shortcut_above_90_percent": not task_reports["MD"][
                "shortcut_risk_above_90_percent"
            ],
        },
    }


def write_shortcut_audit(records_path: str | Path, output_path: str | Path) -> dict:
    records = [json.loads(line) for line in Path(records_path).read_text().splitlines()]
    report = build_shortcut_audit(records)
    report["gate"]["passed"] = all(report["gate"].values())
    Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _public_features(row: Mapping[str, Any], names: Sequence[str]) -> list[float]:
    observation = row["observation"]
    output = []
    for name in names:
        if name == "day":
            output.append(float(observation["day"]))
        elif name == "society_size":
            output.append(float(observation["society_size"]))
        else:
            parts = name.split(".")
            if len(parts) == 2:
                group, feature = parts
                output.append(float(observation[group][feature]))
            else:
                group, index, feature = parts
                values = observation[group]
                output.append(
                    float(values[int(index)][feature]) if int(index) < len(values) else 0.0
                )
    return output


def _trusted_features(row: Mapping[str, Any]) -> list[float]:
    trusted = row["provenance"]["trusted_generation"]
    return [
        float(row["society_size"]),
        float(trusted.get("alpha", 0.0)),
        float(trusted.get("harmful_count", 0.0)),
    ]


def _cross_validated_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    model: str,
    parameter: int,
) -> float:
    predictions = np.empty_like(y)
    for fold in range(5):
        test = folds == fold
        train = ~test
        if model == "tree":
            estimator = _CART(parameter).fit(x[train], y[train])
        elif model == "logistic":
            estimator = _Softmax().fit(x[train], y[train], parameter)
        else:
            estimator = _RandomForest(seed=2718 + fold).fit(x[train], y[train])
        predictions[test] = estimator.predict(x[test])
    return float(np.mean(predictions == y))


def _fold(trajectory_id: str) -> int:
    return int.from_bytes(hashlib.blake2s(trajectory_id.encode(), digest_size=2).digest(), "big") % 5


def _gini(y: np.ndarray) -> float:
    counts = np.asarray(list(Counter(map(int, y)).values()), dtype=float)
    probability = counts / counts.sum()
    return float(1.0 - np.square(probability).sum())
