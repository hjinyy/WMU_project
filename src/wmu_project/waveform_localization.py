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

from .waveform_utils import FAULT_EVENTS, load_ieee30_graph, one_hop_hit


def localization_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in {"CaseName", "EventType", "TargetBus"}]


def preprocess_features(frame: pd.DataFrame) -> np.ndarray:
    cols = localization_feature_columns(frame)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline.fit_transform(frame[cols])


def nearest_neighbor_predict(x: np.ndarray, buses: np.ndarray, event_types: np.ndarray, exclude_self: bool) -> pd.DataFrame:
    rows = []
    for idx in range(len(x)):
        distances = np.linalg.norm(x - x[idx], axis=1)
        if exclude_self:
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


def summarize_localization(result: pd.DataFrame, evaluation: str) -> pd.DataFrame:
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
            "Evaluation": evaluation,
            "EventType": "Overall",
            "exact_bus_accuracy": float(result["ExactMatch"].mean()),
            "top3_accuracy": float(result["Top3Match"].mean()),
            "one_hop_accuracy": float(result["OneHopMatch"].mean()),
        }
    ])
    summary.insert(0, "Evaluation", evaluation)
    return pd.concat([summary, overall], ignore_index=True)


def evaluate_fault_localization(by_case_wide: pd.DataFrame, edge_csv: str | Path, reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    fault_df = by_case_wide.loc[by_case_wide["EventType"].isin(FAULT_EVENTS)].copy().reset_index(drop=True)
    x = preprocess_features(fault_df)
    buses = fault_df["TargetBus"].astype(int).to_numpy()
    event_types = fault_df["EventType"].astype(str).to_numpy()
    graph = load_ieee30_graph(edge_csv)

    debug_frames = []
    summary_frames = []
    for evaluation, exclude_self in [("self_matching", False), ("strict_loo", True)]:
        preds = nearest_neighbor_predict(x, buses, event_types, exclude_self=exclude_self)
        result = fault_df[["CaseName", "EventType", "TargetBus"]].copy()
        result = pd.concat([result, preds], axis=1)
        result["Evaluation"] = evaluation
        result["ExactMatch"] = result["TargetBus"].astype(int) == result["PredictedBus"].astype(int)
        result["Top3Match"] = result.apply(lambda row: int(row["TargetBus"]) in ast.literal_eval(row["Top3Buses"]), axis=1)
        result["OneHopMatch"] = result.apply(lambda row: one_hop_hit(graph, int(row["TargetBus"]), int(row["PredictedBus"])), axis=1)
        result["is_1hop"] = result["OneHopMatch"]
        result = result.rename(columns={"Top3Buses": "Top3Pred", "PredictedBus": "PredBus"})
        debug_frames.append(result)
        summary_frames.append(summarize_localization(result.rename(columns={"PredBus": "PredictedBus", "Top3Pred": "Top3Buses"}), evaluation))

    debug_df = pd.concat(debug_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df.to_csv(reports_dir / "fault_localization_preliminary.csv", index=False)
    debug_df.to_csv(reports_dir / "fault_localization_debug.csv", index=False)

    notes = (
        "Self-matching is a sanity-check separability score because each case can match itself. "
        "Strict LOO excludes the current case and is the more relevant preliminary localization result. "
        "Because each bus currently has only one SLG and one three-phase case, strict LOO remains a low-sample fingerprint test rather than a train/test localization benchmark.\n"
    )
    (reports_dir / "fault_localization_notes.txt").write_text(notes, encoding="utf-8")

    strict_df = debug_df.loc[debug_df["Evaluation"] == "strict_loo"].copy()
    labels = sorted(strict_df["TargetBus"].astype(int).unique())
    cm = confusion_matrix(strict_df["TargetBus"].astype(int), strict_df["PredBus"].astype(int), labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Predicted bus")
    ax.set_ylabel("True bus")
    ax.set_title("Fault localization strict LOO confusion matrix")
    ticks = range(len(labels))
    ax.set_xticks(ticks, labels, rotation=90)
    ax.set_yticks(ticks, labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / "fault_localization_confusion_matrix.png", dpi=200)
    plt.close(fig)
    return summary_df
