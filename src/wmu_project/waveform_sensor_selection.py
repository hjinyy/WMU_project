from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .waveform_utils import FAULT_EVENTS

RANDOM_SEED = 42


@dataclass
class SensorSelectionArtifacts:
    curve: pd.DataFrame
    selected: pd.DataFrame


def subset_columns(frame: pd.DataFrame, buses: list[int]) -> list[str]:
    allowed = {f"Bus{bus:02d}__" for bus in buses}
    base_cols = {"CaseName", "EventType", "TargetBus"}
    return [col for col in frame.columns if col in base_cols or any(col.startswith(prefix) for prefix in allowed)]


def loo_nearest_neighbor_predict(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diffs = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    return y[nearest]


def evaluate_subset(frame: pd.DataFrame, buses: list[int]) -> dict[str, float]:
    cols = [c for c in subset_columns(frame, buses) if c not in {"CaseName", "EventType", "TargetBus"}]
    x = frame[cols].to_numpy(dtype=float)
    y = frame["EventType"].astype(str).to_numpy()
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x)
    pred = loo_nearest_neighbor_predict(x, y)
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    balanced_accuracy = float(balanced_accuracy_score(y, pred))
    load_mask = y == "LoadSwitch"
    loadswitch_recall = float(np.mean(pred[load_mask] == "LoadSwitch")) if load_mask.any() else np.nan
    truth_fault = np.isin(y, list(FAULT_EVENTS))
    pred_fault = np.isin(pred, list(FAULT_EVENTS))
    tp = int(np.sum(truth_fault & pred_fault))
    fp = int(np.sum(~truth_fault & pred_fault))
    fault_precision = tp / (tp + fp) if tp + fp else 0.0
    normal_mask = y == "Normal"
    normal_far = float(np.mean(pred[normal_mask] != "Normal")) if normal_mask.any() else np.nan
    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "LoadSwitch_recall": loadswitch_recall,
        "Fault_precision": fault_precision,
        "Normal_false_alarm_rate": normal_far,
    }


def greedy_selection(frame: pd.DataFrame, buses: list[int]) -> tuple[list[int], list[dict[str, float]]]:
    selected: list[int] = []
    remaining = list(buses)
    curve_rows: list[dict[str, float]] = []
    for k in range(1, len(buses) + 1):
        candidates = []
        for bus in remaining:
            subset = selected + [bus]
            metrics = evaluate_subset(frame, subset)
            candidates.append((bus, metrics))
        candidates.sort(key=lambda item: (item[1]["macro_f1"], item[1]["LoadSwitch_recall"], item[1]["Fault_precision"], item[1]["balanced_accuracy"], -item[0]), reverse=True)
        best_bus, best_metrics = candidates[0]
        selected.append(best_bus)
        remaining.remove(best_bus)
        curve_rows.append({"k": k, "Method": "feature_aware_greedy", "SelectedBuses": str(selected), **best_metrics})
    return selected, curve_rows


def dv_energy_ranking(frame: pd.DataFrame) -> list[int]:
    buses = sorted({int(col[3:5]) for col in frame.columns if col.startswith("Bus")})
    scores = []
    for bus in buses:
        col = f"Bus{bus:02d}__dV_energy_3ph_max"
        if col in frame.columns:
            scores.append((bus, float(frame[col].mean())))
    scores.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return [bus for bus, _ in scores]


def evaluate_sensor_count(frame: pd.DataFrame, reports_dir: Path, figures_dir: Path, random_trials: int = 50) -> SensorSelectionArtifacts:
    buses = sorted({int(col[3:5]) for col in frame.columns if col.startswith("Bus")})
    selected, greedy_rows = greedy_selection(frame, buses)
    dv_rank = dv_energy_ranking(frame)

    rows = list(greedy_rows)
    selected_rows = [{"k": i + 1, "Method": "feature_aware_greedy", "SelectedBus": bus} for i, bus in enumerate(selected)]

    for k in range(1, len(buses) + 1):
        subset = dv_rank[:k]
        metrics = evaluate_subset(frame, subset)
        rows.append({"k": k, "Method": "dv_energy_greedy", "SelectedBuses": str(subset), **metrics})
        selected_rows.append({"k": k, "Method": "dv_energy_greedy", "SelectedBus": subset[-1]})

    rng = np.random.default_rng(RANDOM_SEED)
    for k in range(1, len(buses) + 1):
        trial_metrics = []
        for _ in range(random_trials):
            subset = sorted(rng.choice(buses, size=k, replace=False).tolist())
            trial_metrics.append(evaluate_subset(frame, subset))
        agg = pd.DataFrame(trial_metrics).mean(numeric_only=True).to_dict()
        rows.append({"k": k, "Method": f"random_mean_{random_trials}", "SelectedBuses": "random", **agg})

    curve = pd.DataFrame(rows).sort_values(["Method", "k"]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows)
    curve.to_csv(reports_dir / "sensor_count_curve.csv", index=False)
    selected_df.to_csv(reports_dir / "selected_wmu_by_k.csv", index=False)

    for metric_name, filename, ylabel in [
        ("macro_f1", "sensor_count_macro_f1.png", "Macro-F1"),
        ("balanced_accuracy", "sensor_count_balanced_accuracy.png", "Balanced accuracy"),
        ("LoadSwitch_recall", "sensor_count_loadswitch_recall.png", "LoadSwitch recall"),
        ("Fault_precision", "sensor_count_fault_precision.png", "Fault precision"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, subset_df in curve.groupby("Method"):
            ax.plot(subset_df["k"], subset_df[metric_name], marker="o", label=method)
        ax.set_xlabel("Selected WMU count (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Sensor count analysis: {ylabel}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=200)
        plt.close(fig)

    return SensorSelectionArtifacts(curve=curve, selected=selected_df)
