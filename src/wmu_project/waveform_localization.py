from __future__ import annotations

from pathlib import Path
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .waveform_utils import FAULT_EVENTS, format_bus, load_ieee30_graph, one_hop_hit


def localization_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in {"CaseName", "EventType", "TargetBus"}]


def preprocess_features(frame: pd.DataFrame) -> np.ndarray:
    cols = localization_feature_columns(frame)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline.fit_transform(frame[cols])


def nearest_neighbor_predict(x: np.ndarray, buses: np.ndarray, event_types: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx in range(len(x)):
        distances = np.linalg.norm(x - x[idx], axis=1)
        distances[idx] = np.inf
        order = np.argsort(distances)
        pred_bus = int(buses[order[0]])
        top3 = [int(buses[j]) for j in order[:3]]
        rows.append(
            {
                "CaseIndex": idx,
                "PredictedBus": pred_bus,
                "Top3Buses": str(top3),
                "NearestEventType": event_types[order[0]],
                "NearestDistance": float(distances[order[0]]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_fault_localization(by_case_wide: pd.DataFrame, edge_csv: str | Path, reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    fault_df = by_case_wide.loc[by_case_wide["EventType"].isin(FAULT_EVENTS)].copy().reset_index(drop=True)
    x = preprocess_features(fault_df)
    buses = fault_df["TargetBus"].astype(int).to_numpy()
    event_types = fault_df["EventType"].astype(str).to_numpy()
    preds = nearest_neighbor_predict(x, buses, event_types)
    graph = load_ieee30_graph(edge_csv)
    result = fault_df[["CaseName", "EventType", "TargetBus"]].copy()
    result = pd.concat([result, preds], axis=1)
    result["ExactMatch"] = result["TargetBus"].astype(int) == result["PredictedBus"].astype(int)
    result["Top3Match"] = result.apply(lambda row: int(row["TargetBus"]) in ast.literal_eval(row["Top3Buses"]), axis=1)
    result["OneHopMatch"] = result.apply(lambda row: one_hop_hit(graph, int(row["TargetBus"]), int(row["PredictedBus"])), axis=1)
    summary = (
        result.groupby("EventType", as_index=False)
        .agg(
            exact_bus_accuracy=("ExactMatch", "mean"),
            top3_accuracy=("Top3Match", "mean"),
            one_hop_accuracy=("OneHopMatch", "mean"),
        )
    )
    overall = pd.DataFrame([
        {
            "EventType": "Overall",
            "exact_bus_accuracy": float(result["ExactMatch"].mean()),
            "top3_accuracy": float(result["Top3Match"].mean()),
            "one_hop_accuracy": float(result["OneHopMatch"].mean()),
        }
    ])
    output = pd.concat([summary, overall], ignore_index=True)
    output.to_csv(reports_dir / "fault_localization_preliminary.csv", index=False)

    notes = (
        "Preliminary separability analysis only. Each bus currently has one SLG case and one three-phase case, "
        "so leave-one-out nearest-neighbor localization can rely on the opposite fault type at the same bus as the only same-bus reference. "
        "Reported exact/top-3/1-hop scores should therefore be interpreted as optimistic bus-fingerprint separability, not a fully independent localization benchmark.\n"
    )
    (reports_dir / "fault_localization_notes.txt").write_text(notes, encoding="utf-8")

    labels = sorted(result["TargetBus"].astype(int).unique())
    cm = confusion_matrix(result["TargetBus"].astype(int), result["PredictedBus"].astype(int), labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Predicted bus")
    ax.set_ylabel("True bus")
    ax.set_title("Fault localization preliminary confusion matrix")
    ticks = range(len(labels))
    ax.set_xticks(ticks, labels, rotation=90)
    ax.set_yticks(ticks, labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / "fault_localization_confusion_matrix.png", dpi=200)
    plt.close(fig)
    return output
