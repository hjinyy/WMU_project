from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .waveform_features import select_feature_columns_by_group
from .waveform_utils import EVENT_CLASS_ORDER, FAULT_EVENTS

RANDOM_SEED = 42


@dataclass
class EvaluationArtifacts:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    stratified_metrics: pd.DataFrame
    feature_columns: dict[str, list[str]] | None = None


def build_models() -> dict[str, Pipeline]:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    tree_numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return {
        "LogisticRegression": Pipeline([
            ("prep", numeric),
            ("model", OneVsRestClassifier(LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=RANDOM_SEED))),
        ]),
        "DecisionTree": Pipeline([
            ("prep", tree_numeric),
            ("model", DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight="balanced", max_depth=6)),
        ]),
        "RandomForest": Pipeline([
            ("prep", tree_numeric),
            ("model", RandomForestClassifier(n_estimators=400, random_state=RANDOM_SEED, class_weight="balanced_subsample", n_jobs=-1)),
        ]),
        "LinearSVC": Pipeline([
            ("prep", numeric),
            ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, dual="auto")),
        ]),
    }


def loo_predict(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    loo = LeaveOneOut()
    preds = np.empty_like(y)
    for train_idx, test_idx in loo.split(x, y):
        fitted = clone(model)
        fitted.fit(x.iloc[train_idx], y[train_idx])
        preds[test_idx[0]] = fitted.predict(x.iloc[test_idx])[0]
    return preds


def stratified_predict(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, int] | tuple[None, int]:
    unique, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if counts.size else 0
    if len(unique) < 2 or min_count < 2:
        return None, 0
    n_splits = min(5, min_count)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    preds = np.empty_like(y)
    for train_idx, test_idx in skf.split(x, y):
        fitted = clone(model)
        fitted.fit(x.iloc[train_idx], y[train_idx])
        preds[test_idx] = fitted.predict(x.iloc[test_idx])
    return preds, n_splits


def compute_metrics(truth: np.ndarray, pred: np.ndarray, labels: list[str], model_name: str, feature_group: str, validation: str) -> dict[str, object]:
    per_class = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
    precision, recall, f1, support = per_class
    row: dict[str, object] = {
        "FeatureGroup": feature_group,
        "Model": model_name,
        "Validation": validation,
        "accuracy": accuracy_score(truth, pred),
        "macro_f1": f1_score(truth, pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
    }
    for idx, label in enumerate(labels):
        row[f"precision_{label}"] = precision[idx]
        row[f"recall_{label}"] = recall[idx]
        row[f"f1_{label}"] = f1[idx]
        row[f"support_{label}"] = support[idx]
    truth_fault = np.isin(truth, list(FAULT_EVENTS))
    pred_fault = np.isin(pred, list(FAULT_EVENTS))
    tp = int(np.sum(truth_fault & pred_fault))
    fp = int(np.sum(~truth_fault & pred_fault))
    row["Fault_precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    normal_mask = truth == "Normal"
    row["Normal_false_alarm_rate"] = float(np.mean(pred[normal_mask] != "Normal")) if normal_mask.any() else np.nan
    row["LoadSwitch_recall"] = row.get("recall_LoadSwitch", np.nan)
    misclassified = [case for case, t, p in zip(range(len(truth)), truth, pred) if t == "LoadSwitch" and p in FAULT_EVENTS]
    row["LoadSwitch_to_Fault_count"] = len(misclassified)
    return row


def evaluate_models_for_group(by_case_wide: pd.DataFrame, feature_group: str, models: dict[str, Pipeline] | None = None, selected_models: list[str] | None = None) -> EvaluationArtifacts:
    models = models or build_models()
    labels = [label for label in EVENT_CLASS_ORDER if label in by_case_wide["EventType"].unique()]
    feature_columns = select_feature_columns_by_group(by_case_wide, feature_group)
    x = by_case_wide[feature_columns].copy()
    y = by_case_wide["EventType"].astype(str).to_numpy()
    metrics_rows = []
    pred_rows = []
    strat_rows = []
    active_models = selected_models or list(models)
    for model_name in active_models:
        model = models[model_name]
        pred = loo_predict(model, x, y)
        metrics_rows.append(compute_metrics(y, pred, labels, model_name, feature_group, "LOO"))
        pred_df = by_case_wide[["CaseName", "EventType", "TargetBus"]].copy()
        pred_df["Model"] = model_name
        pred_df["FeatureGroup"] = feature_group
        pred_df["PredictedEventType"] = pred
        pred_df["IsCorrect"] = pred_df["EventType"] == pred_df["PredictedEventType"]
        pred_df["LoadSwitchToFault"] = (pred_df["EventType"] == "LoadSwitch") & pred_df["PredictedEventType"].isin(FAULT_EVENTS)
        pred_rows.append(pred_df)

        skf_pred, n_splits = stratified_predict(model, x, y)
        if skf_pred is not None:
            strat_rows.append(compute_metrics(y, skf_pred, labels, model_name, feature_group, f"StratifiedKFold_{n_splits}"))

    return EvaluationArtifacts(
        metrics=pd.DataFrame(metrics_rows).sort_values(["macro_f1", "balanced_accuracy"], ascending=False).reset_index(drop=True),
        predictions=pd.concat(pred_rows, ignore_index=True),
        stratified_metrics=pd.DataFrame(strat_rows),
        feature_columns={feature_group: feature_columns},
    )


def write_confusion_matrix_csv(truth: np.ndarray, pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    cm = confusion_matrix(truth, pred, labels=labels)
    frame = pd.DataFrame(cm, index=labels, columns=labels)
    frame.index.name = "True"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)


def plot_confusion_matrix(truth: np.ndarray, pred: np.ndarray, labels: list[str], title: str, output_path: Path) -> None:
    cm = confusion_matrix(truth, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_classification_report(metrics: pd.DataFrame, predictions: pd.DataFrame, stratified_metrics: pd.DataFrame, output_path: Path) -> None:
    best = metrics.iloc[0]
    best_mask = (predictions["Model"] == best["Model"]) & (predictions["FeatureGroup"] == best["FeatureGroup"])
    fault_cases = predictions.loc[best_mask & predictions["LoadSwitchToFault"], ["CaseName", "PredictedEventType"]]
    lines = [
        "Waveform event classification report",
        "",
        f"Best LOO model: {best['Model']} on {best['FeatureGroup']}",
        f"macro-F1={best['macro_f1']:.4f}",
        f"balanced_accuracy={best['balanced_accuracy']:.4f}",
        f"LoadSwitch_recall={best['LoadSwitch_recall']:.4f}",
        f"Fault_precision={best['Fault_precision']:.4f}",
        f"Normal_false_alarm_rate={best['Normal_false_alarm_rate']:.4f}",
        "",
        "LoadSwitch -> Fault misclassifications:",
    ]
    if fault_cases.empty:
        lines.append("- none")
    else:
        for _, row in fault_cases.iterrows():
            lines.append(f"- {row['CaseName']} -> {row['PredictedEventType']}")
    if not stratified_metrics.empty:
        lines.extend(["", "Supplementary StratifiedKFold results:", stratified_metrics.to_string(index=False)])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_full_classification(by_case_wide: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> EvaluationArtifacts:
    artifacts = evaluate_models_for_group(by_case_wide, "All_features")
    artifacts.metrics.to_csv(reports_dir / "classification_full_wmu_metrics.csv", index=False)
    write_classification_report(artifacts.metrics, artifacts.predictions, artifacts.stratified_metrics, reports_dir / "classification_full_wmu_report.txt")
    best = artifacts.metrics.iloc[0]
    best_pred = artifacts.predictions.loc[(artifacts.predictions["Model"] == best["Model"]) & (artifacts.predictions["FeatureGroup"] == best["FeatureGroup"])]
    labels = [label for label in EVENT_CLASS_ORDER if label in by_case_wide["EventType"].unique()]
    write_confusion_matrix_csv(
        best_pred["EventType"].to_numpy(),
        best_pred["PredictedEventType"].to_numpy(),
        labels,
        reports_dir / "confusion_matrix_full_wmu.csv",
    )
    best_pred.loc[~best_pred["IsCorrect"]].to_csv(reports_dir / "misclassified_cases_full_wmu.csv", index=False)
    plot_confusion_matrix(best_pred["EventType"].to_numpy(), best_pred["PredictedEventType"].to_numpy(), labels, f"Full WMU confusion matrix ({best['Model']})", figures_dir / "confusion_matrix_full_wmu.png")
    return artifacts


def evaluate_feature_ablation(by_case_wide: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> EvaluationArtifacts:
    models = build_models()
    selected_models = ["LogisticRegression", "RandomForest"]
    metrics_frames = []
    pred_frames = []
    strat_frames = []
    feature_columns_by_group: dict[str, list[str]] = {}
    for group_name in ["DV_energy_only", "Voltage_time_only", "Voltage_time_freq", "Voltage_current", "Voltage_current_unbalance_sequence", "Impedance_added", "All_features"]:
        artifacts = evaluate_models_for_group(by_case_wide, group_name, models=models, selected_models=selected_models)
        metrics_frames.append(artifacts.metrics)
        pred_frames.append(artifacts.predictions)
        feature_columns_by_group[group_name] = artifacts.feature_columns[group_name] if artifacts.feature_columns else []
        if not artifacts.stratified_metrics.empty:
            strat_frames.append(artifacts.stratified_metrics)
    metrics = pd.concat(metrics_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    stratified = pd.concat(strat_frames, ignore_index=True) if strat_frames else pd.DataFrame()
    metrics.to_csv(reports_dir / "feature_ablation_metrics.csv", index=False)
    used_columns_lines = []
    for group_name, columns in feature_columns_by_group.items():
        used_columns_lines.append(f"[{group_name}]")
        used_columns_lines.append(f"count={len(columns)}")
        used_columns_lines.extend(columns)
        used_columns_lines.append("")
    fingerprints = {
        group_name: hashlib.md5("\n".join(feature_columns_by_group[group_name]).encode("utf-8")).hexdigest()
        for group_name in feature_columns_by_group
    }
    used_columns_lines.append("[matrix_fingerprints]")
    for group_name, fingerprint in fingerprints.items():
        used_columns_lines.append(f"{group_name}={fingerprint}")
    (reports_dir / "feature_ablation_used_columns.txt").write_text("\n".join(used_columns_lines) + "\n", encoding="utf-8")

    best_by_group = metrics.sort_values(["FeatureGroup", "macro_f1", "LoadSwitch_recall"], ascending=[True, False, False]).groupby("FeatureGroup", as_index=False).first()
    for metric_name, filename, ylabel in [
        ("macro_f1", "feature_ablation_macro_f1.png", "Macro-F1"),
        ("LoadSwitch_recall", "feature_ablation_loadswitch_recall.png", "LoadSwitch recall"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(best_by_group))
        ax.bar(positions, best_by_group[metric_name], color="#4472c4")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Feature ablation: best model per group ({ylabel})")
        ax.set_xticks(positions, best_by_group["FeatureGroup"], rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=200)
        plt.close(fig)
    return EvaluationArtifacts(metrics=metrics, predictions=predictions, stratified_metrics=stratified, feature_columns=feature_columns_by_group)
