from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(
    "/mnt/c/Users/user/Documents/MATLAB/WMU_final/"
    "WMU_batch_data_ibr_background"
)
DEFAULT_REPORTS = (
    ROOT / "results" / "waveform_ibr_background_diagnostics" / "reports"
)
NONFAULT = {"SSO_Normal", "SSO_LoadSwitch"}
FAULT = {"SSO_SLG_Fault", "SSO_ThreePhase_Fault"}
ID_COLUMNS = {
    "CaseName",
    "EventType",
    "TargetBus",
    "ObservedBus",
    "EventTime",
    "SamplingRateHz",
}
FAULT_FEATURES = [
    "dV_energy_3ph_max",
    "dI_energy_3ph_max",
    "max_sag",
    "I_rms_jump_3ph_max_abs",
    "relative_I_rms_jump",
    "I0_ratio",
    "I2_ratio",
    "V0_ratio",
    "V2_ratio",
    "Delta_I_unbalance_abs",
    "Delta_V_unbalance_abs",
    "Delta_Z_app_abs",
    "Z_drop_ratio",
]
LOADSWITCH_PHASE_FEATURES = [
    "dV_E72_ratio_A",
    "dV_E72_ratio_B",
    "dV_E72_ratio_C",
]
LOADSWITCH_PENALTY = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--max-k", type=int, default=4)
    return parser.parse_args()


def binary_label(event_type: str) -> int:
    if event_type in NONFAULT:
        return 0
    if event_type in FAULT:
        return 1
    raise ValueError(f"Unexpected event type: {event_type}")


def write_inventory(data_dir: Path, reports: Path) -> None:
    files = [
        data_dir / "feature_table_by_bus.csv",
        data_dir / "feature_table_by_case.csv",
        data_dir / "feature_table_by_case_wide.csv",
        data_dir / "dataset_metadata.csv",
    ]
    rows = []
    for path in files:
        frame = pd.read_csv(path)
        rows.append(
            {
                "FileName": path.name,
                "Path": str(path),
                "Rows": len(frame),
                "Columns": len(frame.columns),
                "CaseIdColumn": "CaseName" if "CaseName" in frame else "",
                "EventTypeColumn": "EventType" if "EventType" in frame else "",
                "BusColumns": "|".join(
                    column
                    for column in ("TargetBus", "ObservedBus")
                    if column in frame
                ),
                "EventTypes": "|".join(
                    sorted(frame["EventType"].dropna().astype(str).unique())
                )
                if "EventType" in frame
                else "",
                "ColumnNames": "|".join(frame.columns),
                "Bytes": path.stat().st_size,
                "Usage": (
                    "primary bus-level features"
                    if path.name == "feature_table_by_bus.csv"
                    else "inventory/reference only"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        reports / "hard_constraint_input_inventory.csv", index=False
    )


def write_label_mapping(reports: Path) -> None:
    pd.DataFrame(
        [
            {
                "EventType": event,
                "BinaryFaultLabel": int(event in FAULT),
                "BinaryClass": "Fault" if event in FAULT else "Non-fault",
                "HardConstraintRole": {
                    "SSO_Normal": "Normal false positive must be zero",
                    "SSO_LoadSwitch": "LoadSwitch false positive must be zero",
                    "SSO_SLG_Fault": "SLG false negative must be zero",
                    "SSO_ThreePhase_Fault": (
                        "ThreePhase false negative must be zero"
                    ),
                }[event],
            }
            for event in [
                "SSO_Normal",
                "SSO_LoadSwitch",
                "SSO_SLG_Fault",
                "SSO_ThreePhase_Fault",
            ]
        ]
    ).to_csv(reports / "binary_label_mapping.csv", index=False)


def feature_mapping() -> pd.DataFrame:
    rows = [
        ("Fault severity", "dV_energy", "dV_energy_3ph_max", "direct", True),
        ("Fault severity", "dI_energy", "dI_energy_3ph_max", "direct", True),
        ("Fault severity", "voltage sag", "max_sag", "direct", True),
        ("Fault severity", "V_rms_drop", "max_sag", "closest available proxy", True),
        (
            "Fault severity",
            "I_rms_jump",
            "I_rms_jump_3ph_max",
            "absolute value",
            True,
        ),
        (
            "Fault severity",
            "short-window transient energy",
            "dV_energy_3ph_max|dI_energy_3ph_max",
            "existing event-window cycle-difference RMS energy",
            True,
        ),
        ("Sequence/unbalance", "I0_ratio", "I0_ratio", "direct", True),
        ("Sequence/unbalance", "I2_ratio", "I2_ratio", "direct", True),
        ("Sequence/unbalance", "V0_ratio", "V0_ratio", "direct", True),
        ("Sequence/unbalance", "V2_ratio", "V2_ratio", "direct", True),
        (
            "Sequence/unbalance",
            "Delta_I_unbalance",
            "Delta_I_unbalance",
            "absolute value",
            True,
        ),
        (
            "Sequence/unbalance",
            "Delta_V_unbalance",
            "Delta_V_unbalance",
            "absolute value",
            True,
        ),
        ("Impedance", "Z_app_pre", "Z_app_pre", "direct", False),
        ("Impedance", "Z_app_event/post", "Z_app_post", "direct", False),
        (
            "Impedance",
            "Delta_Z_app",
            "Delta_Z_app",
            "absolute value",
            True,
        ),
        ("Impedance", "Z_drop_ratio", "Z_drop_ratio", "direct", True),
        (
            "LoadSwitch rejection",
            "dV_E72_ratio",
            "dV_E72_ratio_A|dV_E72_ratio_B|dV_E72_ratio_C",
            "minimum normalized phase ratio; balanced 67-77 Hz evidence",
            True,
        ),
        (
            "LoadSwitch rejection",
            "60-80 Hz dV ratio",
            "dV_E72_ratio_A|dV_E72_ratio_B|dV_E72_ratio_C",
            "nearest existing band is 67-77 Hz",
            True,
        ),
        (
            "LoadSwitch rejection",
            "67-77 Hz dV ratio",
            "dV_E72_ratio_A|dV_E72_ratio_B|dV_E72_ratio_C",
            "direct existing band",
            True,
        ),
        (
            "LoadSwitch rejection",
            "70-80 Hz dV ratio",
            "dV_E72_ratio_A|dV_E72_ratio_B|dV_E72_ratio_C",
            "nearest existing band is 67-77 Hz",
            True,
        ),
        (
            "LoadSwitch rejection",
            "I_rms_post - I_rms_pre",
            "I_rms_jump_A|I_rms_jump_B|I_rms_jump_C",
            "direct phase differences",
            False,
        ),
        (
            "LoadSwitch rejection",
            "relative I_rms jump",
            "I_rms_jump_[A-C] / I_rms_pre_[A-C]",
            "derived from existing table",
            True,
        ),
        (
            "LoadSwitch rejection",
            "switching-window energy",
            "dV_energy_3ph_max|dI_energy_3ph_max",
            "existing event-window energy proxy",
            False,
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "FeatureGroup",
            "RequestedFeature",
            "MappedColumns",
            "MappingOrTransform",
            "UsedInFixedFaultScore",
        ],
    )


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source, target in [
        ("I_rms_jump_3ph_max", "I_rms_jump_3ph_max_abs"),
        ("Delta_I_unbalance", "Delta_I_unbalance_abs"),
        ("Delta_V_unbalance", "Delta_V_unbalance_abs"),
        ("Delta_Z_app", "Delta_Z_app_abs"),
    ]:
        out[target] = pd.to_numeric(out[source], errors="coerce").abs()
    relative = []
    for phase in ("A", "B", "C"):
        jump = pd.to_numeric(out[f"I_rms_jump_{phase}"], errors="coerce").abs()
        pre = pd.to_numeric(out[f"I_rms_pre_{phase}"], errors="coerce").abs()
        relative.append(jump / pre.clip(lower=1e-12))
    out["relative_I_rms_jump"] = pd.concat(relative, axis=1).max(axis=1)
    out["BinaryFaultLabel"] = out["EventType"].map(binary_label)
    return out


def normalize_by_bus(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame[
        ["CaseName", "EventType", "TargetBus", "ObservedBus", "BinaryFaultLabel"]
    ].copy()
    columns = FAULT_FEATURES + LOADSWITCH_PHASE_FEATURES
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        normalized[column] = np.nan
        for bus, index in frame.groupby("ObservedBus").groups.items():
            bus_values = values.loc[index]
            low = float(bus_values.quantile(0.05))
            high = float(bus_values.quantile(0.95))
            scaled = (bus_values - low) / max(high - low, 1e-12)
            normalized.loc[index, column] = scaled.clip(0.0, 1.0)
    return normalized


def build_bus_scores(
    normalized: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
    score_rows = []
    fault_vectors: dict[int, np.ndarray] = {}
    load_vectors: dict[int, np.ndarray] = {}
    for bus, group in normalized.groupby("ObservedBus"):
        group = group.sort_values("CaseName").reset_index(drop=True)
        fault_evidence = group[FAULT_FEATURES].max(axis=1).to_numpy(float)
        # A LoadSwitch excites the 67-77 Hz ratio on all phases; the minimum
        # phase score rejects events where only one phase is high.
        load_evidence = group[LOADSWITCH_PHASE_FEATURES].min(axis=1).to_numpy(float)
        score = fault_evidence - LOADSWITCH_PENALTY * load_evidence
        fault_vectors[int(bus)] = fault_evidence
        load_vectors[int(bus)] = load_evidence
        for idx, row in group.iterrows():
            score_rows.append(
                {
                    "CaseName": row["CaseName"],
                    "EventType": row["EventType"],
                    "TargetBus": row["TargetBus"],
                    "ObservedBus": int(bus),
                    "BinaryFaultLabel": int(row["BinaryFaultLabel"]),
                    "FaultEvidence": fault_evidence[idx],
                    "LoadSwitchEvidence": load_evidence[idx],
                    "FaultScore": score[idx],
                }
            )
    return pd.DataFrame(score_rows), fault_vectors, load_vectors


def best_threshold(score: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    nonfault_max = float(score[labels == 0].max())
    fault_min = float(score[labels == 1].min())
    margin = fault_min - nonfault_max
    if margin > 0:
        return (nonfault_max + fault_min) / 2.0, margin
    candidates = np.unique(score)
    candidates = np.r_[
        candidates[0] - 1e-9,
        (candidates[:-1] + candidates[1:]) / 2.0,
        candidates[-1] + 1e-9,
    ]
    best = None
    for threshold in candidates:
        pred = (score > threshold).astype(int)
        errors = int(np.sum(pred != labels))
        fp = int(np.sum((labels == 0) & (pred == 1)))
        fn = int(np.sum((labels == 1) & (pred == 0)))
        key = (errors, fp + fn, abs(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold))
    assert best is not None
    return best[1], margin


def metric_row(
    subset: tuple[int, ...],
    score: np.ndarray,
    case_frame: pd.DataFrame,
) -> dict[str, object]:
    labels = case_frame["BinaryFaultLabel"].to_numpy(int)
    threshold, margin = best_threshold(score, labels)
    pred = (score > threshold).astype(int)
    event = case_frame["EventType"].to_numpy(str)
    normal_fp = int(np.sum((event == "SSO_Normal") & (pred == 1)))
    load_fp = int(np.sum((event == "SSO_LoadSwitch") & (pred == 1)))
    slg_fn = int(np.sum((event == "SSO_SLG_Fault") & (pred == 0)))
    three_fn = int(np.sum((event == "SSO_ThreePhase_Fault") & (pred == 0)))
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    feasible = normal_fp == load_fp == slg_fn == three_fn == 0
    return {
        "WMUSet": "|".join(map(str, subset)),
        "k": len(subset),
        "NormalFP": normal_fp,
        "LoadSwitchFP": load_fp,
        "SLGFN": slg_fn,
        "ThreePhaseFN": three_fn,
        "TotalFP": int(fp),
        "TotalFN": int(fn),
        "FaultRecall": tp / (tp + fn),
        "NonFaultSpecificity": tn / (tn + fp),
        "LoadSwitchSpecificity": 1.0
        - load_fp / max(int(np.sum(event == "SSO_LoadSwitch")), 1),
        "BinaryAccuracy": accuracy_score(labels, pred),
        "Feasible": feasible,
        "FaultScoreMargin": margin,
        "BestThreshold": threshold,
        "NonFaultMaxScore": float(score[labels == 0].max()),
        "FaultMinScore": float(score[labels == 1].min()),
        "ClassifierType": "fixed physical threshold rule",
        "UsedFeatureSet": (
            "max normalized fault severity/sequence/impedance evidence "
            "- 0.5 * balanced 67-77Hz LoadSwitch evidence"
        ),
    }


def exhaustive_search(
    fault_vectors: dict[int, np.ndarray],
    load_vectors: dict[int, np.ndarray],
    case_frame: pd.DataFrame,
    max_k: int,
) -> pd.DataFrame:
    rows = []
    buses = sorted(fault_vectors)
    for k in range(1, max_k + 1):
        for subset in itertools.combinations(buses, k):
            fault_score = np.max(
                np.stack([fault_vectors[bus] for bus in subset]), axis=0
            )
            load_score = np.max(
                np.stack([load_vectors[bus] for bus in subset]), axis=0
            )
            score = fault_score - LOADSWITCH_PENALTY * load_score
            rows.append(metric_row(subset, score, case_frame))
    return pd.DataFrame(rows)


def build_selected_matrix(
    frame: pd.DataFrame, buses: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        column
        for column in frame.columns
        if column not in ID_COLUMNS | {"BinaryFaultLabel"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows = []
    for (case, event, target), group in frame.loc[
        frame["ObservedBus"].isin(buses)
    ].groupby(["CaseName", "EventType", "TargetBus"], dropna=False):
        row = {
            "CaseName": case,
            "EventType": event,
            "TargetBus": target,
            "BinaryFaultLabel": binary_label(str(event)),
        }
        for bus in buses:
            bus_row = group.loc[group["ObservedBus"] == bus].iloc[0]
            for feature in features:
                row[f"Bus{bus:02d}__{feature}"] = bus_row[feature]
        rows.append(row)
    matrix = pd.DataFrame(rows).sort_values("CaseName").reset_index(drop=True)
    return matrix[[c for c in matrix if not c.startswith("Bus")]], matrix[
        [c for c in matrix if c.startswith("Bus")]
    ]


def models() -> dict[str, Pipeline]:
    scaled = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "LogisticRegression": Pipeline(
            scaled
            + [
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000, class_weight="balanced", random_state=42
                    ),
                )
            ]
        ),
        "LinearSVM": Pipeline(
            scaled
            + [
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=42,
                        dual="auto",
                        max_iter=5000,
                    ),
                )
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def predict_loo(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(y)
    for train, test in LeaveOneOut().split(x):
        fitted = clone(model).fit(x.iloc[train], y[train])
        pred[test] = fitted.predict(x.iloc[test])
    return pred


def predict_stratified(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(y)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in folds.split(x, y):
        fitted = clone(model).fit(x.iloc[train], y[train])
        pred[test] = fitted.predict(x.iloc[test])
    return pred


def predict_location_out(
    model: Pipeline, meta: pd.DataFrame, x: pd.DataFrame, y: np.ndarray
) -> np.ndarray:
    groups = []
    normal_counter = 0
    for row in meta.itertuples(index=False):
        if pd.notna(row.TargetBus):
            groups.append(f"Bus{int(row.TargetBus):02d}")
        else:
            normal_counter += 1
            groups.append(f"Normal{normal_counter:02d}")
    groups = np.asarray(groups)
    pred = np.zeros_like(y)
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        fitted = clone(model).fit(x.iloc[train], y[train])
        pred[test] = fitted.predict(x.iloc[test])
    return pred


def ml_metrics(
    selected: tuple[int, ...], frame: pd.DataFrame
) -> pd.DataFrame:
    meta, x = build_selected_matrix(frame, selected)
    y = meta["BinaryFaultLabel"].to_numpy(int)
    rows = []
    for model_name, model in models().items():
        validations = {
            "LeaveOneCaseOut": predict_loo(model, x, y),
            "LeaveTargetLocationOut": predict_location_out(model, meta, x, y),
            "Stratified3Fold": predict_stratified(model, x, y),
        }
        for validation, pred in validations.items():
            event = meta["EventType"].to_numpy(str)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "WMUSet": "|".join(map(str, selected)),
                    "Model": model_name,
                    "Validation": validation,
                    "NormalFP": int(
                        np.sum((event == "SSO_Normal") & (pred == 1))
                    ),
                    "LoadSwitchFP": int(
                        np.sum((event == "SSO_LoadSwitch") & (pred == 1))
                    ),
                    "SLGFN": int(
                        np.sum((event == "SSO_SLG_Fault") & (pred == 0))
                    ),
                    "ThreePhaseFN": int(
                        np.sum(
                            (event == "SSO_ThreePhase_Fault") & (pred == 0)
                        )
                    ),
                    "TotalFP": int(fp),
                    "TotalFN": int(fn),
                    "FaultRecall": tp / (tp + fn),
                    "NonFaultSpecificity": tn / (tn + fp),
                    "LoadSwitchSpecificity": 1.0
                    - np.mean(pred[event == "SSO_LoadSwitch"] == 1),
                    "BinaryAccuracy": accuracy_score(y, pred),
                    "HardConstraintFeasible": bool(fp == 0 and fn == 0),
                    "Notes": (
                        "Location-out groups use TargetBus for fault and "
                        "LoadSwitch cases; each Normal case is its own group."
                        if validation == "LeaveTargetLocationOut"
                        else (
                            "Only three Normal cases; fold estimate has high variance."
                            if validation == "Stratified3Fold"
                            else ""
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    write_inventory(args.data_dir, args.reports_dir)
    write_label_mapping(args.reports_dir)
    mapping = feature_mapping()
    mapping.to_csv(
        args.reports_dir / "hard_constraint_feature_mapping.csv", index=False
    )

    by_bus = add_derived_features(
        pd.read_csv(args.data_dir / "feature_table_by_bus.csv")
    )
    normalized = normalize_by_bus(by_bus)
    score_cases, fault_vectors, load_vectors = build_bus_scores(normalized)
    score_cases.to_csv(
        args.reports_dir / "hard_constraint_fault_scores_by_case_bus.csv",
        index=False,
    )
    case_frame = (
        score_cases.loc[score_cases["ObservedBus"] == 1]
        .sort_values("CaseName")
        .reset_index(drop=True)
    )
    exhaustive = exhaustive_search(
        fault_vectors, load_vectors, case_frame, args.max_k
    )
    exhaustive.to_csv(
        args.reports_dir / "exhaustive_subset_hard_constraint_results.csv",
        index=False,
    )
    singles = exhaustive.loc[exhaustive["k"] == 1].copy()
    singles.to_csv(
        args.reports_dir / "single_wmu_hard_constraint_results.csv", index=False
    )
    singles[
        [
            "WMUSet",
            "FaultScoreMargin",
            "BestThreshold",
            "NonFaultMaxScore",
            "FaultMinScore",
            "Feasible",
        ]
    ].to_csv(
        args.reports_dir / "fault_score_margin_by_single_wmu.csv", index=False
    )

    feasible = exhaustive.loc[exhaustive["Feasible"]].copy()
    min_k = int(feasible["k"].min())
    minimum = feasible.loc[feasible["k"] == min_k].sort_values(
        ["FaultScoreMargin", "WMUSet"], ascending=[False, True]
    )
    minimum.to_csv(
        args.reports_dir / "feasible_minimum_wmu_sets.csv", index=False
    )
    selected = minimum.head(1).copy()
    selected.to_csv(
        args.reports_dir / "selected_minimum_wmu_set.csv", index=False
    )
    selected_set = tuple(map(int, selected.iloc[0]["WMUSet"].split("|")))
    selected_fault_score = np.max(
        np.stack([fault_vectors[bus] for bus in selected_set]), axis=0
    )
    selected_load_score = np.max(
        np.stack([load_vectors[bus] for bus in selected_set]), axis=0
    )
    selected_score = (
        selected_fault_score - LOADSWITCH_PENALTY * selected_load_score
    )
    threshold = float(selected.iloc[0]["BestThreshold"])
    predictions = case_frame[
        ["CaseName", "EventType", "TargetBus", "BinaryFaultLabel"]
    ].copy()
    predictions["FaultEvidence"] = selected_fault_score
    predictions["LoadSwitchEvidence"] = selected_load_score
    predictions["FaultScore"] = selected_score
    predictions["Threshold"] = threshold
    predictions["PredictedBinaryFaultLabel"] = (
        selected_score > threshold
    ).astype(int)
    predictions["Correct"] = (
        predictions["BinaryFaultLabel"]
        == predictions["PredictedBinaryFaultLabel"]
    )
    predictions.to_csv(
        args.reports_dir / "selected_minimum_wmu_case_predictions.csv",
        index=False,
    )
    ml_metrics(selected_set, by_bus).to_csv(
        args.reports_dir / "ml_binary_classifier_hard_constraint_results.csv",
        index=False,
    )
    counts = (
        exhaustive.groupby("k")["Feasible"]
        .agg(TotalSubsets="size", FeasibleSubsetCount="sum")
        .reset_index()
    )
    counts.to_csv(
        args.reports_dir / "hard_constraint_feasible_counts_by_k.csv",
        index=False,
    )
    print(counts.to_string(index=False))
    print("\nSelected:")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
