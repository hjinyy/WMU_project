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
ID_COLUMNS = {"CaseName", "EventType", "TargetBus", "ObservedBus", "EventTime", "SamplingRateHz"}


@dataclass
class SensorSelectionArtifacts:
    curve: pd.DataFrame
    selected: pd.DataFrame
    debug: pd.DataFrame


def _feature_columns(by_bus: pd.DataFrame) -> list[str]:
    return [col for col in by_bus.columns if col not in ID_COLUMNS]


def build_selected_bus_case_matrix(by_bus: pd.DataFrame, buses: list[int]) -> pd.DataFrame:
    subset = by_bus.loc[by_bus["ObservedBus"].isin(buses)].copy()
    features = _feature_columns(subset)
    rows = []
    for (case_name, event_type, target_bus), group in subset.groupby(["CaseName", "EventType", "TargetBus"], dropna=False):
        row = {
            "CaseName": case_name,
            "EventType": event_type,
            "TargetBus": target_bus,
        }
        observed = sorted(group["ObservedBus"].astype(int).unique())
        for bus in buses:
            bus_row = group.loc[group["ObservedBus"] == bus]
            if bus_row.empty:
                for feature in features:
                    row[f"Bus{bus:02d}__{feature}"] = np.nan
                continue
            bus_row = bus_row.iloc[0]
            for feature in features:
                row[f"Bus{bus:02d}__{feature}"] = bus_row[feature]
        row["SelectedBusCount"] = len(observed)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["EventType", "TargetBus", "CaseName"]).reset_index(drop=True)


def loo_nearest_neighbor_predict(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diffs = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    return y[nearest]


def evaluate_subset(by_bus: pd.DataFrame, buses: list[int]) -> tuple[dict[str, float], dict[str, object]]:
    buses = [int(bus) for bus in buses]
    matrix = build_selected_bus_case_matrix(by_bus, buses)
    feature_cols = [col for col in matrix.columns if col.startswith("Bus")]
    x = matrix[feature_cols].to_numpy(dtype=float)
    y = matrix["EventType"].astype(str).to_numpy()
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
    metrics = {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "LoadSwitch_recall": loadswitch_recall,
        "Fault_precision": fault_precision,
        "Normal_false_alarm_rate": normal_far,
    }
    debug = {
        "selected_buses": str(buses),
        "num_features": len(feature_cols),
        "feature_columns": "|".join(feature_cols),
        **metrics,
    }
    return metrics, debug


def greedy_selection(by_bus: pd.DataFrame, buses: list[int]) -> tuple[list[int], list[dict[str, float]], list[dict[str, object]]]:
    selected: list[int] = []
    remaining = [int(bus) for bus in buses]
    curve_rows: list[dict[str, float]] = []
    debug_rows: list[dict[str, object]] = []
    for k in range(1, len(buses) + 1):
        candidates = []
        for bus in remaining:
            subset = selected + [bus]
            metrics, debug = evaluate_subset(by_bus, subset)
            debug.update({"Method": "feature_aware_greedy_candidate", "k": k, "CandidateBus": bus})
            debug_rows.append(debug)
            candidates.append((bus, metrics, debug))
        candidates.sort(key=lambda item: (item[1]["macro_f1"], item[1]["LoadSwitch_recall"], item[1]["Fault_precision"], item[1]["balanced_accuracy"], -item[0]), reverse=True)
        best_bus, best_metrics, best_debug = candidates[0]
        selected.append(best_bus)
        remaining.remove(best_bus)
        curve_rows.append({"k": k, "Method": "feature_aware_greedy", "SelectedBuses": str(selected), **best_metrics})
        best_debug = dict(best_debug)
        best_debug.update({"Method": "feature_aware_greedy_selected", "k": k, "CandidateBus": best_bus})
        debug_rows.append(best_debug)
    return selected, curve_rows, debug_rows


def dv_energy_ranking(by_bus: pd.DataFrame) -> list[int]:
    scores = (
        by_bus.groupby("ObservedBus", as_index=False)["dV_energy_3ph_max"]
        .mean()
        .sort_values(["dV_energy_3ph_max", "ObservedBus"], ascending=[False, True])
    )
    return scores["ObservedBus"].astype(int).tolist()


def evaluate_sensor_count(by_bus: pd.DataFrame, reports_dir: Path, figures_dir: Path, random_trials: int = 50) -> SensorSelectionArtifacts:
    buses = sorted(int(bus) for bus in by_bus["ObservedBus"].astype(int).unique())
    selected, greedy_rows, debug_rows = greedy_selection(by_bus, buses)
    dv_rank = dv_energy_ranking(by_bus)

    rows = list(greedy_rows)
    selected_rows = [{"k": i + 1, "Method": "feature_aware_greedy", "SelectedBus": bus} for i, bus in enumerate(selected)]

    for k in range(1, len(buses) + 1):
        subset = dv_rank[:k]
        metrics, debug = evaluate_subset(by_bus, subset)
        rows.append({"k": k, "Method": "dv_energy_greedy", "SelectedBuses": str(subset), **metrics})
        debug.update({"Method": "dv_energy_greedy", "k": k, "CandidateBus": subset[-1]})
        debug_rows.append(debug)
        selected_rows.append({"k": k, "Method": "dv_energy_greedy", "SelectedBus": subset[-1]})

    rng = np.random.default_rng(RANDOM_SEED)
    for k in range(1, len(buses) + 1):
        trial_metrics = []
        for trial in range(random_trials):
            subset = sorted(rng.choice(buses, size=k, replace=False).tolist())
            metrics, debug = evaluate_subset(by_bus, subset)
            debug.update({"Method": "random_trial", "k": k, "CandidateBus": -1, "Trial": trial})
            debug_rows.append(debug)
            trial_metrics.append(metrics)
        agg = pd.DataFrame(trial_metrics).mean(numeric_only=True).to_dict()
        rows.append({"k": k, "Method": f"random_mean_{random_trials}", "SelectedBuses": "random", **agg})

    curve = pd.DataFrame(rows).sort_values(["Method", "k"]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows)
    debug_df = pd.DataFrame(debug_rows)
    curve.to_csv(reports_dir / "sensor_count_curve.csv", index=False)
    selected_df.to_csv(reports_dir / "selected_wmu_by_k.csv", index=False)
    debug_df.to_csv(reports_dir / "sensor_selection_debug.csv", index=False)

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

    return SensorSelectionArtifacts(curve=curve, selected=selected_df, debug=debug_df)
