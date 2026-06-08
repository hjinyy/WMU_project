from __future__ import annotations

import argparse
import ast
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = [
    "SSO_Normal",
    "SSO_LoadSwitch",
    "SSO_SLG_Fault",
    "SSO_ThreePhase_Fault",
]
PALETTE = dict(zip(CLASS_ORDER, ["#4c78a8", "#f58518", "#e45756", "#72b7b2"]))
META = {
    "CaseName",
    "EventType",
    "TargetBus",
    "ObservedBus",
    "EventTime",
    "SamplingRateHz",
    "ObservedBusCount",
    "star_bus_dV",
    "star_bus_dI",
}
SEED = 42


FEATURE_REQUESTS = {
    "max_dv_energy": ["dV_energy_3ph_max", "max_dV_energy"],
    "mean_dv_energy": ["dV_energy_3ph_mean"],
    "max_sag": ["max_sag"],
    "min_voltage_rms": [],
    "voltage_rms_drop": ["max_sag", "mean_sag"],
    "max_di_energy": ["dI_energy_3ph_max", "max_dI_energy"],
    "mean_di_energy": ["dI_energy_3ph_mean"],
    "max_current_rms": ["I_rms_post_A", "I_rms_post_B", "I_rms_post_C"],
    "current_rms_rise": ["I_rms_jump_3ph_max"],
    "band_energy_20_30Hz": ["dV_SSO20_30_ratio_A", "dV_SSO20_30_ratio_B", "dV_SSO20_30_ratio_C"],
    "band_energy_5_55Hz": ["dV_SSC5_55_ratio_A", "dV_SSC5_55_ratio_B", "dV_SSC5_55_ratio_C"],
    "res_ratio": ["dV_Res_ratio_3ph_max"],
    "hf_ratio": ["dV_HF_ratio_3ph_max"],
    "sso_band_ratio": ["dV_SSO20_30_ratio_A", "dV_SSO20_30_ratio_B", "dV_SSO20_30_ratio_C"],
    "dominant_freq": [],
    "dominant_freq_power": [],
    "V0_over_V1": ["V0_ratio"],
    "V2_over_V1": ["V2_ratio"],
    "I0_over_I1": ["I0_ratio"],
    "I2_over_I1": ["I2_ratio"],
    "negative_sequence_ratio": ["V2_ratio", "I2_ratio"],
    "zero_sequence_ratio": ["V0_ratio", "I0_ratio"],
    "Z_app_pre": ["Z_app_pre"],
    "Z_app_event": ["Z_app_post"],
    "delta_Z_app": ["Delta_Z_app"],
    "impedance_drop_ratio": ["Z_drop_ratio"],
}

PLOT_GROUPS = {
    "feature_distribution_dv_energy.png": [
        "dV_energy_3ph_max",
        "dV_energy_3ph_mean",
    ],
    "feature_distribution_di_energy.png": [
        "dI_energy_3ph_max",
        "dI_energy_3ph_mean",
        "I_rms_jump_3ph_max",
    ],
    "feature_distribution_sag.png": ["max_sag", "mean_sag"],
    "feature_distribution_sso_band_energy.png": [
        "dV_SSO20_30_ratio_A",
        "dV_SSC5_55_ratio_A",
        "dV_Res_ratio_3ph_max",
        "dV_HF_ratio_3ph_max",
    ],
    "feature_distribution_sequence_unbalance.png": [
        "V0_ratio",
        "V2_ratio",
        "I0_ratio",
        "I2_ratio",
        "Delta_V_unbalance",
        "Delta_I_unbalance",
    ],
    "feature_distribution_impedance.png": [
        "Z_app_pre",
        "Z_app_post",
        "Delta_Z_app",
        "Z_drop_ratio",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results" / "waveform_ibr_background_diagnostics"),
    )
    parser.add_argument("--source-results-dir", required=True)
    return parser.parse_args()


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=1,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in META and pd.api.types.is_numeric_dtype(frame[col])
    ]


def write_inventory(
    data_dir: Path, source_results: Path, reports: Path
) -> pd.DataFrame:
    raw_dir = data_dir.parent / "WMU_batch_raw_ibr_background"
    candidates = [
        data_dir / "feature_table_by_bus.csv",
        data_dir / "feature_table_by_case.csv",
        data_dir / "feature_table_by_case_wide.csv",
        raw_dir / "dataset_metadata.csv",
        source_results / "reports" / "dataset_case_index.csv",
        source_results / "reports" / "classification_full_wmu_metrics.csv",
        source_results / "reports" / "sensor_count_curve.csv",
        source_results / "reports" / "fault_localization_preliminary.csv",
        source_results / "reports" / "fault_localization_debug.csv",
    ]
    rows = []
    for path in candidates:
        exists = path.exists()
        shape = (0, 0)
        note = "missing"
        if exists:
            frame = pd.read_csv(path)
            shape = frame.shape
            note = "reused; source file left unchanged"
        rows.append(
            {
                "FileName": path.name,
                "Path": str(path),
                "Exists": exists,
                "Rows": shape[0],
                "Columns": shape[1],
                "Notes": note,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(reports / "input_data_inventory.csv", index=False)
    return out


def feature_mapping(by_bus: pd.DataFrame, reports: Path) -> pd.DataFrame:
    rows = []
    columns = set(by_bus.columns)
    for requested, candidates in FEATURE_REQUESTS.items():
        matches = [candidate for candidate in candidates if candidate in columns]
        rows.append(
            {
                "RequestedFeature": requested,
                "MappedColumns": "|".join(matches),
                "MappingStatus": "mapped" if matches else "not_available",
                "Aggregation": "case maximum across observed buses",
                "Notes": (
                    "Closest available extracted feature(s)"
                    if matches
                    else "Not present in existing feature tables; raw waveforms were not reprocessed."
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(reports / "feature_name_mapping.csv", index=False)
    return out


def case_feature_frame(by_bus: pd.DataFrame) -> pd.DataFrame:
    selected = sorted(
        {
            col
            for cols in PLOT_GROUPS.values()
            for col in cols
            if col in by_bus.columns
        }
    )
    maxima = (
        by_bus.groupby(["CaseName", "EventType", "TargetBus"], dropna=False)[selected]
        .max()
        .reset_index()
    )
    return maxima


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def distributions(case_features: pd.DataFrame, reports: Path, figures: Path) -> None:
    summary_rows = []
    rank_rows = []
    for feature in numeric_columns(case_features):
        groups = []
        for label in CLASS_ORDER:
            values = (
                pd.to_numeric(
                    case_features.loc[case_features["EventType"] == label, feature],
                    errors="coerce",
                )
                .dropna()
                .to_numpy()
            )
            groups.append(values)
            if len(values):
                summary_rows.append(
                    {
                        "Feature": feature,
                        "EventType": label,
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
                        "median": np.median(values),
                        "q25": np.quantile(values, 0.25),
                        "q75": np.quantile(values, 0.75),
                        "IQR": np.quantile(values, 0.75) - np.quantile(values, 0.25),
                    }
                )
        valid = [group for group in groups if len(group)]
        _, pvalue = kruskal(*valid) if len(valid) > 1 else (np.nan, np.nan)
        effects = []
        for i, j in itertools.combinations(range(len(CLASS_ORDER)), 2):
            delta = cliffs_delta(groups[i], groups[j])
            effects.append((abs(delta), delta, CLASS_ORDER[i], CLASS_ORDER[j]))
        effects.sort(reverse=True)
        best = effects[0]
        pairwise = "; ".join(
            f"{left} vs {right}:{signed:.6g}"
            for _, signed, left, right in effects
        )
        rank_rows.append(
            {
                "Feature": feature,
                "KruskalWallisPValue": pvalue,
                "MaxAbsCliffsDelta": best[0],
                "TopSeparatingClassPair": f"{best[2]} vs {best[3]}",
                "CliffsDeltaSigned": best[1],
                "PairwiseCliffsDelta": pairwise,
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        reports / "feature_distribution_summary.csv", index=False
    )
    ranking = pd.DataFrame(rank_rows).sort_values(
        ["MaxAbsCliffsDelta", "KruskalWallisPValue"], ascending=[False, True]
    )
    ranking.to_csv(reports / "feature_separability_ranking.csv", index=False)

    for filename, requested_features in PLOT_GROUPS.items():
        features = [feature for feature in requested_features if feature in case_features]
        ncols = 2
        nrows = math.ceil(len(features) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(14, 4.8 * nrows), squeeze=False
        )
        for ax, feature in zip(axes.flat, features):
            plot = case_features[["EventType", feature]].dropna()
            sns.boxplot(
                data=plot,
                x="EventType",
                y=feature,
                order=CLASS_ORDER,
                hue="EventType",
                palette=PALETTE,
                legend=False,
                ax=ax,
                fliersize=0,
            )
            sns.stripplot(
                data=plot,
                x="EventType",
                y=feature,
                order=CLASS_ORDER,
                color="black",
                alpha=0.42,
                size=3,
                jitter=0.18,
                ax=ax,
            )
            values = np.abs(pd.to_numeric(plot[feature], errors="coerce"))
            if values.max() > 0 and values.max() / max(values[values > 0].min(), 1e-15) > 1e4:
                ax.set_yscale("symlog", linthresh=max(values.quantile(0.1), 1e-12))
                ax.set_title(f"{feature} (symlog)")
            else:
                ax.set_title(feature)
            ax.tick_params(axis="x", rotation=24)
            ax.set_xlabel("")
        for ax in axes.flat[len(features) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=210, bbox_inches="tight")
        plt.close(fig)


def labels_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    precision, recall, _, _ = precision_recall_fscore_support(
        truth, pred, labels=CLASS_ORDER, zero_division=0
    )
    fault_truth = np.isin(truth, CLASS_ORDER[2:])
    fault_pred = np.isin(pred, CLASS_ORDER[2:])
    tp = np.sum(fault_truth & fault_pred)
    fp = np.sum(~fault_truth & fault_pred)
    return {
        "macro_f1": f1_score(
            truth, pred, labels=CLASS_ORDER, average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "normal_recall": recall[0],
        "LoadSwitch_recall": recall[1],
        "fault_precision": tp / (tp + fp) if tp + fp else 0.0,
        "SLG_recall": recall[2],
        "ThreePhase_recall": recall[3],
    }


def loo_predictions(x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for train, test in LeaveOneOut().split(x):
        fitted = clone(model()).fit(x.iloc[train], y[train])
        pred[test[0]] = fitted.predict(x.iloc[test])[0]
    return pred


def stratified_predictions(x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for train, test in cv.split(x, y):
        fitted = clone(model()).fit(x.iloc[train], y[train])
        pred[test] = fitted.predict(x.iloc[test])
    return pred


def importance(by_case_wide: pd.DataFrame, reports: Path, figures: Path) -> None:
    feature_cols = numeric_columns(by_case_wide)
    x = by_case_wide[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = by_case_wide["EventType"].astype(str).to_numpy()
    fitted = model().fit(x, y)
    values = fitted.named_steps["model"].feature_importances_
    impurity = (
        pd.DataFrame({"Feature": feature_cols, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    impurity.to_csv(reports / "feature_importance_randomforest.csv", index=False)
    plot_bar(
        impurity.head(20),
        "Importance",
        figures / "feature_importance_randomforest_top20.png",
        "RandomForest impurity importance: top 20 full-WMU features",
    )

    candidates = impurity.head(60)["Feature"].tolist()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fold_rows = []
    for fold, (train, test) in enumerate(cv.split(x, y), start=1):
        fitted_fold = model().fit(x.iloc[train][candidates], y[train])
        perm = permutation_importance(
            fitted_fold,
            x.iloc[test][candidates],
            y[test],
            scoring="f1_macro",
            n_repeats=12,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        for feature, mean, std in zip(
            candidates, perm.importances_mean, perm.importances_std
        ):
            fold_rows.append(
                {
                    "Fold": fold,
                    "Feature": feature,
                    "PermutationImportance": mean,
                    "PermutationStd": std,
                }
            )
    perm_frame = (
        pd.DataFrame(fold_rows)
        .groupby("Feature", as_index=False)
        .agg(
            PermutationImportance=("PermutationImportance", "mean"),
            PermutationStd=("PermutationImportance", "std"),
        )
        .sort_values("PermutationImportance", ascending=False)
    )
    perm_frame["CandidateScope"] = "top 60 impurity-ranked full-WMU features"
    perm_frame.to_csv(reports / "permutation_importance.csv", index=False)
    plot_bar(
        perm_frame.head(20),
        "PermutationImportance",
        figures / "permutation_importance_top20.png",
        "Cross-validated permutation importance: top 20",
    )

    top_union = impurity.head(35)["Feature"].tolist()
    class_rows = []
    for label in CLASS_ORDER:
        binary = (y == label).astype(int)
        binary_model = model().fit(x[top_union], binary)
        for feature, value in zip(
            top_union, binary_model.named_steps["model"].feature_importances_
        ):
            class_rows.append(
                {"EventType": label, "Feature": feature, "Importance": value}
            )
    class_frame = pd.DataFrame(class_rows)
    class_frame.to_csv(reports / "classwise_feature_importance.csv", index=False)
    matrix = class_frame.pivot(
        index="EventType", columns="Feature", values="Importance"
    ).reindex(CLASS_ORDER)
    keep = matrix.max(axis=0).sort_values(ascending=False).head(20).index
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(matrix[keep], cmap="viridis", ax=ax)
    ax.set_title("One-vs-rest RandomForest feature importance")
    ax.set_xlabel("Full-WMU feature")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(
        figures / "classwise_feature_importance_heatmap.png",
        dpi=210,
        bbox_inches="tight",
    )
    plt.close(fig)

    dv_feature = next(
        (
            feature
            for feature in feature_cols
            if feature.endswith("__dV_energy_3ph_max")
        ),
        "max_dV_energy",
    )
    pred = loo_predictions(x[[dv_feature]], y)
    pd.DataFrame(
        [
            {
                "Feature": dv_feature,
                "Validation": "LOO",
                **labels_metrics(y, pred),
            }
        ]
    ).to_csv(reports / "single_feature_dv_energy_performance.csv", index=False)


def plot_bar(frame: pd.DataFrame, value: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ordered = frame.sort_values(value)
    ax.barh(ordered["Feature"], ordered[value], color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(value)
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def bus_columns(frame: pd.DataFrame, buses: list[int]) -> list[str]:
    prefixes = tuple(f"Bus{bus:02d}__" for bus in buses)
    return [col for col in frame.columns if col.startswith(prefixes)]


def bus_eval(by_case_wide: pd.DataFrame, buses: list[int]) -> tuple[dict, np.ndarray]:
    cols = bus_columns(by_case_wide, buses)
    y = by_case_wide["EventType"].astype(str).to_numpy()
    pred = stratified_predictions(by_case_wide[cols], y)
    return labels_metrics(y, pred), pred


def grouped_predictions(
    by_case_wide: pd.DataFrame, buses: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    cols = bus_columns(by_case_wide, buses)
    y = by_case_wide["EventType"].astype(str).to_numpy()
    target = pd.to_numeric(by_case_wide["TargetBus"], errors="coerce")
    normals = by_case_wide.index[by_case_wide["EventType"] == "SSO_Normal"].tolist()
    truth, predictions = [], []
    for bus in range(1, 31):
        test = by_case_wide.index[target == bus].tolist()
        if bus <= len(normals):
            test.append(normals[bus - 1])
        if not test:
            continue
        train = [idx for idx in by_case_wide.index if idx not in test]
        fitted = model().fit(by_case_wide.loc[train, cols], y[train])
        predictions.extend(fitted.predict(by_case_wide.loc[test, cols]).tolist())
        truth.extend(y[test].tolist())
    return np.asarray(truth), np.asarray(predictions)


def sensor_diagnostics(
    by_bus: pd.DataFrame,
    by_case_wide: pd.DataFrame,
    source_results: Path,
    reports: Path,
    figures: Path,
) -> None:
    rows = []
    recalls = []
    dv_scores = []
    y = by_case_wide["EventType"].astype(str).to_numpy()
    for bus in range(1, 31):
        metrics, pred = bus_eval(by_case_wide, [bus])
        cm = confusion_matrix(y, pred, labels=CLASS_ORDER)
        rows.append(
            {
                "Bus": bus,
                "Validation": "stratified_3fold",
                **metrics,
                "ConfusionMatrix": np.array2string(cm, separator=","),
            }
        )
        recalls.append(
            {
                "Bus": bus,
                **{
                    label: precision_recall_fscore_support(
                        y, pred, labels=[label], zero_division=0
                    )[1][0]
                    for label in CLASS_ORDER
                },
            }
        )
        dv_col = f"Bus{bus:02d}__dV_energy_3ph_max"
        dv_pred = stratified_predictions(by_case_wide[[dv_col]], y)
        dv_scores.append(
            {"Bus": bus, "macro_f1": labels_metrics(y, dv_pred)["macro_f1"]}
        )
    performance = pd.DataFrame(rows).sort_values("Bus")
    performance.to_csv(reports / "single_wmu_performance_by_bus.csv", index=False)
    plot_bus_metric(
        performance,
        "macro_f1",
        figures / "single_wmu_macro_f1_by_bus.png",
        "Single-WMU stratified 3-fold macro-F1 by bus",
    )
    plot_bus_metric(
        performance,
        "balanced_accuracy",
        figures / "single_wmu_balanced_accuracy_by_bus.png",
        "Single-WMU stratified 3-fold balanced accuracy by bus",
    )
    recall_frame = pd.DataFrame(recalls).set_index("Bus")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(recall_frame, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Single-WMU class recall")
    fig.tight_layout()
    fig.savefig(
        figures / "single_wmu_class_recall_heatmap.png", dpi=210, bbox_inches="tight"
    )
    plt.close(fig)

    compare_features = [
        "dV_energy_3ph_max",
        "dI_energy_3ph_max",
        "max_sag",
        "dV_SSO20_30_ratio_A",
        "I0_ratio",
        "Z_drop_ratio",
    ]
    compare = by_bus.loc[by_bus["ObservedBus"].isin([1, 5, 13, 30])].copy()
    diagnostics = (
        compare.groupby("ObservedBus")[compare_features]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    diagnostics.columns = [
        "_".join(str(part) for part in col if str(part))
        for col in diagnostics.columns.to_flat_index()
    ]
    all_row = {
        "ObservedBus": "AllBusMean",
        **{
            f"{feature}_{stat}": getattr(by_bus[feature], stat)()
            for feature in compare_features
            for stat in ["mean", "median", "std"]
        },
    }
    diagnostics = pd.concat([diagnostics, pd.DataFrame([all_row])], ignore_index=True)
    diagnostics.to_csv(reports / "selected_bus1_diagnostics.csv", index=False)
    normalized = (
        by_bus.groupby("ObservedBus")[compare_features].median()
        / by_bus[compare_features].median().replace(0, np.nan)
    ).loc[[1, 5, 13, 30]]
    fig, ax = plt.subplots(figsize=(12, 6))
    normalized.T.plot(kind="bar", ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Median / all-bus median (log scale)")
    ax.set_title("Bus 1 versus comparison WMUs")
    ax.legend(title="Observed bus")
    fig.tight_layout()
    fig.savefig(
        figures / "bus1_vs_other_bus_feature_comparison.png",
        dpi=210,
        bbox_inches="tight",
    )
    plt.close(fig)

    feature_rank = performance.sort_values(
        ["macro_f1", "balanced_accuracy", "Bus"], ascending=[False, False, True]
    )["Bus"].tolist()
    dv_rank = (
        pd.DataFrame(dv_scores)
        .sort_values(["macro_f1", "Bus"], ascending=[False, True])["Bus"]
        .tolist()
    )
    rng = np.random.default_rng(SEED)
    extended = []
    grouped_rows = []
    existing_curve = pd.read_csv(source_results / "reports" / "sensor_count_curve.csv")
    for _, row in existing_curve.loc[
        existing_curve["Method"].isin(["feature_aware_greedy", "dv_energy_greedy"])
        & (existing_curve["k"] <= 10)
    ].iterrows():
        extended.append(
            {
                "k": int(row["k"]),
                "Method": (
                    "feature-aware greedy (existing LOO)"
                    if row["Method"] == "feature_aware_greedy"
                    else "dv_energy greedy (existing LOO)"
                ),
                "SelectedBuses": row["SelectedBuses"],
                "macro_f1": row["macro_f1"],
                "balanced_accuracy": row["balanced_accuracy"],
                "normal_recall": row["Normal_recall"],
                "LoadSwitch_recall": row["LoadSwitch_recall"],
                "fault_precision": row["Fault_precision"],
                "SLG_recall": np.nan,
                "ThreePhase_recall": np.nan,
            }
        )
    for k in range(1, 11):
        selected = feature_rank[:k]
        metrics, _ = bus_eval(by_case_wide, selected)
        extended.append(
            {
                "k": k,
                "Method": "diagnostic single-bus rank",
                "SelectedBuses": str(selected),
                **metrics,
            }
        )
        dv_selected = dv_rank[:k]
        metrics, _ = bus_eval(by_case_wide, dv_selected)
        extended.append(
            {
                "k": k,
                "Method": "diagnostic dv-energy rank",
                "SelectedBuses": str(dv_selected),
                **metrics,
            }
        )
        truth, pred = grouped_predictions(by_case_wide, selected)
        grouped_metric = labels_metrics(truth, pred)
        grouped_rows.append(
            {
                "Validation": "leave-target-location-out",
                "k": k,
                "SelectedBuses": str(selected),
                **grouped_metric,
                "Notes": "Each located event is held out with all events at the same target bus; the 3 Normal cases are each tested once in folds 1-3.",
            }
        )
        extended.append(
            {
                "k": k,
                "Method": "grouped CV feature-aware rank",
                "SelectedBuses": str(selected),
                **grouped_metric,
            }
        )
        random_metrics = []
        for _ in range(5):
            choice = sorted(rng.choice(np.arange(1, 31), size=k, replace=False).tolist())
            metric, _ = bus_eval(by_case_wide, choice)
            random_metrics.append(metric)
        for stat in ["mean", "min", "max"]:
            record = {"k": k, "Method": f"random {stat}", "SelectedBuses": ""}
            for key in random_metrics[0]:
                values = [item[key] for item in random_metrics]
                record[key] = getattr(np, stat)(values)
            extended.append(record)
        best = performance.iloc[performance["macro_f1"].argmax()]
        worst = performance.iloc[performance["macro_f1"].argmin()]
        for name, source in [("single best", best), ("single worst", worst)]:
            extended.append(
                {
                    "k": k,
                    "Method": name,
                    "SelectedBuses": f"[{int(source['Bus'])}]",
                    **{key: source[key] for key in labels_metrics(y, y)},
                }
            )
    extended_frame = pd.DataFrame(extended)
    extended_frame.to_csv(reports / "sensor_count_extended_summary.csv", index=False)
    grouped_frame = pd.DataFrame(grouped_rows)
    grouped_frame.to_csv(reports / "grouped_cv_classification_summary.csv", index=False)
    plot_sensor(
        extended_frame,
        "macro_f1",
        figures / "sensor_count_macro_f1_extended.png",
        "Extended sensor-count macro-F1",
    )
    plot_sensor(
        extended_frame,
        "balanced_accuracy",
        figures / "sensor_count_balanced_accuracy_extended.png",
        "Extended sensor-count balanced accuracy",
    )
    plot_sensor(
        extended_frame,
        "LoadSwitch_recall",
        figures / "sensor_count_class_recall_extended.png",
        "Extended sensor-count LoadSwitch recall",
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(grouped_frame["k"], grouped_frame["macro_f1"], marker="o")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Sensor count")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Leave-target-location-out macro-F1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        figures / "grouped_cv_macro_f1_by_sensor_count.png",
        dpi=210,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_bus_metric(frame: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(frame["Bus"], frame[metric], color="#4c78a8")
    ax.set_xticks(range(1, 31))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Observed bus")
    ax.set_ylabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_sensor(frame: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    methods = [
        "feature-aware greedy (existing LOO)",
        "dv_energy greedy (existing LOO)",
        "diagnostic single-bus rank",
        "diagnostic dv-energy rank",
        "grouped CV feature-aware rank",
        "random mean",
        "random min",
        "random max",
        "single best",
        "single worst",
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    for method in methods:
        subset = frame[frame["Method"] == method]
        ax.plot(subset["k"], subset[metric], marker="o", label=method)
    finite = pd.to_numeric(frame[metric], errors="coerce").dropna()
    lower = max(0.0, float(finite.min()) - 0.025) if not finite.empty else 0.0
    ax.set_ylim(lower, 1.005)
    ax.set_xlabel("Sensor count")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def localization(
    source_results: Path, reports: Path, figures: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    debug = pd.read_csv(source_results / "reports" / "fault_localization_debug.csv")
    strict = debug[debug["Evaluation"] == "strict_loo"].copy()
    strict["Top3List"] = strict["Top3Pred"].apply(ast.literal_eval)
    edges = pd.read_csv(ROOT / "data" / "ieee30_edges.csv")
    graph = nx.from_pandas_edgelist(edges, edges.columns[0], edges.columns[1])
    strict["GraphDistance"] = strict.apply(
        lambda row: nx.shortest_path_length(
            graph, int(row["TargetBus"]), int(row["PredBus"])
        ),
        axis=1,
    )
    strict["TwoHopMatch"] = strict["GraphDistance"] <= 2

    communities = list(nx.community.greedy_modularity_communities(graph))
    communities = sorted(communities, key=lambda group: min(group))
    zone_map = {
        int(bus): f"Zone{idx + 1}"
        for idx, group in enumerate(communities)
        for bus in sorted(group)
    }
    zone_rows = [
        {"Bus": bus, "Zone": zone, "Method": "greedy modularity community detection"}
        for bus, zone in sorted(zone_map.items())
    ]
    pd.DataFrame(zone_rows).to_csv(
        reports / "localization_zone_definition.csv", index=False
    )
    strict["TrueZone"] = strict["TargetBus"].astype(int).map(zone_map)
    strict["PredZone"] = strict["PredBus"].astype(int).map(zone_map)
    strict["ZoneMatch"] = strict["TrueZone"] == strict["PredZone"]
    per_case = strict[
        [
            "CaseName",
            "EventType",
            "TargetBus",
            "PredBus",
            "Top3Pred",
            "ExactMatch",
            "Top3Match",
            "OneHopMatch",
            "TwoHopMatch",
            "GraphDistance",
            "TrueZone",
            "PredZone",
            "ZoneMatch",
        ]
    ].copy()
    per_case.to_csv(reports / "localization_per_case_distance.csv", index=False)

    metric_rows = []
    for event_type, subset in [
        ("Overall", strict),
        *list(strict.groupby("EventType")),
    ]:
        metric_rows.append(
            {
                "EventType": event_type,
                "exact_accuracy": subset["ExactMatch"].mean(),
                "top3_accuracy": subset["Top3Match"].mean(),
                "one_hop_accuracy": subset["OneHopMatch"].mean(),
                "two_hop_accuracy": subset["TwoHopMatch"].mean(),
                "mean_graph_distance_error": subset["GraphDistance"].mean(),
                "median_graph_distance_error": subset["GraphDistance"].median(),
                "zone_accuracy": subset["ZoneMatch"].mean(),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(reports / "localization_distance_metrics.csv", index=False)
    metrics[metrics["EventType"] != "Overall"].to_csv(
        reports / "localization_by_fault_type.csv", index=False
    )

    zones = sorted(set(zone_map.values()))
    zone_cm = confusion_matrix(
        strict["TrueZone"], strict["PredZone"], labels=zones, normalize="true"
    )
    zone_pred = strict["PredZone"].to_numpy()
    zone_true = strict["TrueZone"].to_numpy()
    zone_metrics = pd.DataFrame(
        [
            {
                "zone_accuracy": np.mean(zone_true == zone_pred),
                "zone_macro_f1": f1_score(
                    zone_true, zone_pred, labels=zones, average="macro", zero_division=0
                ),
                "zone_count": len(zones),
            }
        ]
    )
    zone_metrics.to_csv(reports / "localization_zone_metrics.csv", index=False)
    pd.DataFrame(zone_cm, index=zones, columns=zones).to_csv(
        reports / "localization_zone_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    distances = np.sort(strict["GraphDistance"].to_numpy())
    ax.step(distances, np.arange(1, len(distances) + 1) / len(distances), where="post")
    ax.set_xlabel("Graph distance error (hops)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Strict-LOO localization graph-distance CDF")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "localization_distance_cdf.png", dpi=210)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(strict["GraphDistance"], discrete=True, ax=ax)
    ax.set_title("Strict-LOO localization graph-distance error")
    fig.tight_layout()
    fig.savefig(figures / "localization_distance_histogram.png", dpi=210)
    plt.close(fig)

    overall = metrics[metrics["EventType"] == "Overall"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["Exact", "Top-3", "1-hop", "2-hop", "Zone"]
    values = [
        overall["exact_accuracy"],
        overall["top3_accuracy"],
        overall["one_hop_accuracy"],
        overall["two_hop_accuracy"],
        overall["zone_accuracy"],
    ]
    ax.bar(names, values, color=["#777777", "#999999", "#4c78a8", "#72b7b2", "#f58518"])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Accuracy")
    ax.set_title("Localization tolerance summary")
    fig.tight_layout()
    fig.savefig(figures / "localization_onehop_twohop_summary.png", dpi=210)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(zone_cm, annot=True, fmt=".2f", cmap="magma", xticklabels=zones, yticklabels=zones, ax=ax)
    ax.set_xlabel("Predicted zone")
    ax.set_ylabel("True zone")
    ax.set_title("Normalized zone confusion matrix")
    fig.tight_layout()
    fig.savefig(figures / "localization_zone_confusion_matrix.png", dpi=210)
    plt.close(fig)

    labels = list(range(1, 31))
    cm = confusion_matrix(
        strict["TargetBus"].astype(int),
        strict["PredBus"].astype(int),
        labels=labels,
        normalize="true",
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, cmap="magma", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Strict-LOO normalized bus confusion matrix")
    ax.set_xlabel("Predicted bus")
    ax.set_ylabel("True bus")
    fig.tight_layout()
    fig.savefig(figures / "localization_normalized_confusion_matrix.png", dpi=210)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    strict["PredBus"].value_counts().sort_index().reindex(labels, fill_value=0).plot.bar(ax=ax)
    ax.set_title("Strict-LOO predicted bus frequency")
    ax.set_xlabel("Predicted bus")
    fig.tight_layout()
    fig.savefig(figures / "localization_predicted_bus_frequency.png", dpi=210)
    plt.close(fig)

    distance_matrix = pd.DataFrame(
        nx.floyd_warshall_numpy(graph, nodelist=labels), index=labels, columns=labels
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(distance_matrix, cmap="viridis", ax=ax)
    ax.scatter(
        [labels.index(int(bus)) + 0.5 for bus in strict["PredBus"]],
        [labels.index(int(bus)) + 0.5 for bus in strict["TargetBus"]],
        marker="x",
        color="red",
        s=24,
        label="strict-LOO pair",
    )
    ax.set_title("IEEE-30 graph-distance matrix with true/predicted pairs")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "localization_graph_distance_heatmap.png", dpi=210)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=strict,
        x="EventType",
        y="GraphDistance",
        order=CLASS_ORDER[2:],
        hue="EventType",
        palette=PALETTE,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=strict,
        x="EventType",
        y="GraphDistance",
        order=CLASS_ORDER[2:],
        color="black",
        alpha=0.45,
        ax=ax,
    )
    ax.set_title("Localization graph distance by fault type")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figures / "localization_distance_by_fault_type.png", dpi=210)
    plt.close(fig)
    return metrics, zone_metrics


def write_readme(
    results: Path,
    data_dir: Path,
    separability: pd.DataFrame,
    importance_frame: pd.DataFrame,
    sensor: pd.DataFrame,
    grouped: pd.DataFrame,
    localization_metrics: pd.DataFrame,
    zone_metrics: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        columns = frame.columns.tolist()
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, row in frame.iterrows():
            values = []
            for column in columns:
                value = row[column]
                values.append(f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    best = sensor.sort_values("macro_f1", ascending=False).iloc[0]
    worst = sensor.sort_values("macro_f1", ascending=True).iloc[0]
    bus1 = sensor[sensor["Bus"] == 1].iloc[0]
    grouped1 = grouped[grouped["k"] == 1].iloc[0]
    loc = localization_metrics[localization_metrics["EventType"] == "Overall"].iloc[0]
    top_sep = markdown_table(
        separability.head(10)[
            ["Feature", "MaxAbsCliffsDelta", "TopSeparatingClassPair"]
        ]
    )
    top_imp = markdown_table(
        importance_frame.head(20)[["Feature", "Importance"]]
    )
    text = f"""# IBR-background waveform diagnostics

## 1. Purpose

The prior 84-case IBR-like SSO background study obtained macro-F1 = 1.0 for event classification, but that headline result did not explain feature-level separability, the immediate k=1 sensor-count saturation, or the weak strict-LOO exact-bus localization. This additive diagnostics pass addresses those gaps without rerunning Simulink or modifying raw waveforms.

## 2. Dataset and reused inputs

- Raw dataset (read-only reference): `{data_dir.parent / 'WMU_batch_raw_ibr_background'}`
- Reused feature tables: `{data_dir}`
- Cases: SSO_Normal 3, SSO_LoadSwitch 21, SSO_SLG_Fault 30, SSO_ThreePhase_Fault 30 (84 total)
- Deterministic conditions: one SSO setting, one 15% load-switch setting, fixed fault parameters, no injected measurement noise.

## 3. Feature distribution and separability

The plots in `figures/feature_distribution_*.png` show individual samples over boxplots, so the three Normal cases remain visible. Symlog is used only when a feature spans more than four orders of magnitude. The strongest univariate separators are:

{top_sep}

`reports/single_feature_dv_energy_performance.csv` directly tests whether one dV-energy feature alone reproduces the four-class LOO result. The ranking should not be read as causal proof: many bus/phase features are correlated because all cases were generated under a narrow deterministic design.

## 4. Feature importance

RandomForest impurity importance was computed on the full wide WMU table. Cross-validated permutation importance was then applied to the top 60 impurity candidates to avoid thousands of low-information permutations on only 84 cases.

{top_imp}

Class-wise one-vs-rest importances are in `classwise_feature_importance.csv`. Voltage disturbance energy mainly exposes event severity; SSO-band ratios help distinguish the persistent background signature; current magnitude/jump features help separate switching from faults; sequence/unbalance features are especially relevant to SLG versus balanced three-phase faults; impedance changes contribute to fault versus switching discrimination.

Several top full-WMU impurity features are pre-event current magnitudes. That is
a warning that operating-point/location fingerprints and correlated predictors
also help the classifier; it strengthens the case for grouped validation and
prevents interpreting the importance ranking as purely event-physics evidence.

## 5. Sensor-count diagnostics

- Best single WMU: bus {int(best.Bus)} (macro-F1 {best.macro_f1:.4f})
- Worst single WMU: bus {int(worst.Bus)} (macro-F1 {worst.macro_f1:.4f})
- Bus 1: macro-F1 {bus1.macro_f1:.4f}, balanced accuracy {bus1.balanced_accuracy:.4f}
- Leave-target-location-out at k=1: macro-F1 {grouped1.macro_f1:.4f}, balanced accuracy {grouped1.balanced_accuracy:.4f}

The k=1 plateau is therefore a property of this deterministic, high-severity, full-network-observation feature design rather than evidence that one WMU is generally sufficient. The grouped split removes every event at the held target location and tests each Normal case once; with only three Normal records, Normal estimates remain high variance. The selected-bus ranking is diagnostic, not a universal placement optimum.

## 6. Fault-localization diagnostics

- Exact: {loc.exact_accuracy:.4f}
- Top-3: {loc.top3_accuracy:.4f}
- 1-hop: {loc.one_hop_accuracy:.4f}
- 2-hop: {loc.two_hop_accuracy:.4f}
- Mean/median graph-distance error: {loc.mean_graph_distance_error:.3f} / {loc.median_graph_distance_error:.3f} hops
- Zone accuracy: {zone_metrics.iloc[0].zone_accuracy:.4f}; zone macro-F1: {zone_metrics.iloc[0].zone_macro_f1:.4f}

Exact-bus localization remains preliminary because each bus has only one case per fault type and strict LOO becomes nearest-neighbor fingerprint transfer across different event instances. Neighbor/zone metrics are more realistic for this dataset, but they do not replace validation across fault resistance, inception angle, SSO amplitude/frequency, operating point, and noise.

## 7. Research interpretation

Event classification succeeds under the current controlled conditions. The feature distributions and importance tables explain why: broad severity, sequence, current, and impedance signatures create strong and partly redundant class separation. Placement is correspondingly easy and saturates early. Localization should be framed as neighborhood/zone screening rather than exact-bus identification until richer repeated conditions are available.

## 8. Limitations and next experiments

- Only 3 Normal cases.
- One SSO condition and one 15% load-switch magnitude.
- Fixed fault resistance/parameters and deterministic waveforms.
- No measurement noise, missing channels, timing jitter, or topology uncertainty.
- Strong feature correlation and many features relative to 84 cases.

Next experiments should vary SSO magnitude/frequency/damping, fault resistance and inception angle, load-switch size, operating point, and measurement noise. Repeated cases per bus are required for a defensible exact-bus localization benchmark.
"""
    (results / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    results = Path(args.results_dir)
    source_results = Path(args.source_results_dir)
    reports = results / "reports"
    figures = results / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    write_inventory(data_dir, source_results, reports)
    by_bus = pd.read_csv(data_dir / "feature_table_by_bus.csv")
    by_case = pd.read_csv(data_dir / "feature_table_by_case.csv")
    by_case_wide = pd.read_csv(data_dir / "feature_table_by_case_wide.csv")
    for frame in [by_bus, by_case, by_case_wide]:
        frame["EventType"] = frame["EventType"].astype(str)
    feature_mapping(by_bus, reports)
    case_features = case_feature_frame(by_bus)
    distributions(case_features, reports, figures)
    importance(by_case_wide, reports, figures)
    sensor_diagnostics(by_bus, by_case_wide, source_results, reports, figures)
    localization_metrics, zone_metrics = localization(
        source_results, reports, figures
    )

    separability = pd.read_csv(reports / "feature_separability_ranking.csv")
    importance_frame = pd.read_csv(reports / "feature_importance_randomforest.csv")
    sensor = pd.read_csv(reports / "single_wmu_performance_by_bus.csv")
    grouped = pd.read_csv(reports / "grouped_cv_classification_summary.csv")
    write_readme(
        results,
        data_dir,
        separability,
        importance_frame,
        sensor,
        grouped,
        localization_metrics,
        zone_metrics,
    )
    print(f"Diagnostics written to {results}")


if __name__ == "__main__":
    main()
