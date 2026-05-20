from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd

from wmu_project.waveform_classification import evaluate_feature_ablation, evaluate_full_classification
from wmu_project.waveform_features import compute_feature_tables, export_feature_tables
from wmu_project.waveform_io import list_cases, output_roots
from wmu_project.waveform_localization import evaluate_fault_localization
from wmu_project.waveform_quality import build_quality_outputs, write_quality_outputs
from wmu_project.waveform_sensor_selection import evaluate_sensor_count
from wmu_project.waveform_utils import build_case_index_frame, to_local_path

REPO_RESULTS = Path("results/waveform_event_analysis")
EXPECTED_COUNTS = {
    "Normal": 3,
    "LoadSwitch": 20,
    "SLG_Fault": 30,
    "ThreePhase_Fault": 30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WMU waveform event analysis pipeline")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--f0", type=float, default=50.0)
    return parser.parse_args()


def archive_existing_outputs(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "reports",
        output_dir / "figures",
        output_dir / "feature_table_by_bus.csv",
        output_dir / "feature_table_by_case.csv",
        output_dir / "feature_table_by_case_wide.csv",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    archive_root = output_dir / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    archive_dir = archive_root / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), archive_dir / path.name)
    return archive_dir


def verify_input_inventory(cases_df: pd.DataFrame) -> None:
    counts = cases_df["EventType"].astype(str).value_counts().to_dict()
    total = len(cases_df)
    if total != 83 or any(counts.get(key, 0) != EXPECTED_COUNTS[key] for key in EXPECTED_COUNTS):
        raise RuntimeError(
            f"Input inventory mismatch: total={total}, counts={counts}, expected_total=83, expected_counts={EXPECTED_COUNTS}"
        )


def write_repo_results(src_reports: Path, src_figures: Path) -> None:
    repo_reports = REPO_RESULTS / "reports"
    repo_figures = REPO_RESULTS / "figures"
    repo_reports.mkdir(parents=True, exist_ok=True)
    repo_figures.mkdir(parents=True, exist_ok=True)
    for name in [
        "data_quality_summary.md",
        "classification_full_wmu_metrics.csv",
        "classification_full_wmu_report.txt",
        "confusion_matrix_full_wmu.csv",
        "misclassified_cases_full_wmu.csv",
        "feature_ablation_metrics.csv",
        "feature_ablation_used_columns.txt",
        "sensor_count_curve.csv",
        "selected_wmu_by_k.csv",
        "fault_localization_preliminary.csv",
        "fault_localization_debug.csv",
        "fault_localization_notes.txt",
        "final_analysis_summary.md",
    ]:
        src = src_reports / name
        if src.exists():
            (repo_reports / name).write_bytes(src.read_bytes())
    for name in [
        "confusion_matrix_full_wmu.png",
        "feature_ablation_macro_f1.png",
        "feature_ablation_loadswitch_recall.png",
        "sensor_count_macro_f1.png",
        "sensor_count_balanced_accuracy.png",
        "sensor_count_loadswitch_recall.png",
        "sensor_count_fault_precision.png",
        "fault_localization_confusion_matrix.png",
    ]:
        src = src_figures / name
        if src.exists():
            (repo_figures / name).write_bytes(src.read_bytes())


def build_final_summary(cases_df: pd.DataFrame, quality_df: pd.DataFrame, feature_tables, cls_metrics: pd.DataFrame, cls_predictions: pd.DataFrame, ablation_metrics: pd.DataFrame, sensor_curve: pd.DataFrame, selected_wmu: pd.DataFrame, localization_summary: pd.DataFrame) -> str:
    status_counts = quality_df["Status"].value_counts().to_dict()
    class_counts = cases_df["EventType"].astype(str).value_counts().to_dict()
    best_cls = cls_metrics.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]
    best_ablation = ablation_metrics.sort_values(["macro_f1", "LoadSwitch_recall"], ascending=False).iloc[0]
    dv_only = ablation_metrics.loc[ablation_metrics["FeatureGroup"] == "DV_energy_only"].sort_values(["macro_f1", "LoadSwitch_recall"], ascending=False).iloc[0]
    all_feat = ablation_metrics.loc[ablation_metrics["FeatureGroup"] == "All_features"].sort_values(["macro_f1", "LoadSwitch_recall"], ascending=False).iloc[0]
    greedy_curve = sensor_curve.loc[sensor_curve["Method"] == "feature_aware_greedy"].sort_values("k")
    max_f1 = float(greedy_curve["macro_f1"].max())
    plateau_threshold = max_f1 - 0.01
    plateau_row = greedy_curve.loc[greedy_curve["macro_f1"] >= plateau_threshold].sort_values("k").iloc[0]
    best_k = int(plateau_row["k"])
    selected_k = selected_wmu.loc[(selected_wmu["Method"] == "feature_aware_greedy") & (selected_wmu["k"] <= best_k), "SelectedBus"].tolist()
    best_model_mask = (cls_predictions["Model"] == best_cls["Model"]) & (cls_predictions["FeatureGroup"] == best_cls["FeatureGroup"])
    ls_fault_cases = cls_predictions.loc[best_model_mask & cls_predictions["LoadSwitchToFault"], ["CaseName", "PredictedEventType"]].drop_duplicates()

    lines = [
        "# Final Waveform Event Analysis Summary",
        "",
        f"- Input raw event files: {len(cases_df)}",
        f"- Class counts: {class_counts}",
        "- LoadSwitch case count is 20 because only buses with existing Pd or Qd were regenerated under the 15% abrupt load-increase condition.",
        f"- Quality status counts: OK={status_counts.get('OK', 0)}, WARNING={status_counts.get('WARNING', 0)}, FAILED={status_counts.get('FAILED', 0)}",
        f"- Generated bus-level feature count: {len([c for c in feature_tables.by_bus.columns if c not in {'CaseName','EventType','TargetBus','ObservedBus','EventTime','SamplingRateHz'}])}",
        f"- Full-WMU best model: {best_cls['Model']} | macro-F1={best_cls['macro_f1']:.4f} | balanced_accuracy={best_cls['balanced_accuracy']:.4f}",
        f"- Best feature ablation group: {best_ablation['FeatureGroup']} ({best_ablation['Model']}) | macro-F1={best_ablation['macro_f1']:.4f}",
        f"- All_features vs DV_energy_only macro-F1 delta: {all_feat['macro_f1'] - dv_only['macro_f1']:+.4f}",
        f"- Sensor-count saturation k (within 0.01 macro-F1 of best greedy result): {best_k}",
        f"- Selected WMU combination up to saturation k: {selected_k}",
        "",
        "## LoadSwitch misclassified as Fault",
    ]
    if ls_fault_cases.empty:
        lines.append("- none")
    else:
        for _, row in ls_fault_cases.iterrows():
            lines.append(f"- {row['CaseName']} -> {row['PredictedEventType']}")
    lines.extend([
        "",
        "## Fault localization preliminary",
        "```",
        localization_summary.to_string(index=False),
        "```",
        "",
        "## Current limitations",
        "- Normal class has only 3 cases.",
        "- LoadSwitch cases currently cover only one disturbance strength: 15% load increase.",
        "- Fault resistance, clearing time, and inception angle are not varied.",
        "- Fault localization remains preliminary because there is only one case per bus per fault type.",
        "",
        "## Next steps",
        "- Expand LoadSwitch intensity levels to 5%, 15%, and 25%.",
        "- Vary fault resistance.",
        "- Vary clearing time and inception angle.",
        "- Build a larger dataset with train/test separation across repeated conditions.",
        "- Extend to weak-grid or DER-integrated operating scenarios.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_dir = to_local_path(args.input_dir)
    output_dir = to_local_path(args.output_dir)

    cases = list_cases(input_dir)
    case_index = build_case_index_frame(cases)
    verify_input_inventory(case_index)

    archive_dir = archive_existing_outputs(output_dir)
    reports_dir, figures_dir = output_roots(output_dir)
    case_index.to_csv(reports_dir / "dataset_case_index.csv", index=False)

    quality_df, quality_summary = build_quality_outputs(cases)
    write_quality_outputs(quality_df, quality_summary, reports_dir, reports_dir / "data_quality_summary.md")

    feature_tables = compute_feature_tables(cases, f0=args.f0)
    export_feature_tables(feature_tables, output_dir)

    full_cls = evaluate_full_classification(feature_tables.by_case_wide, reports_dir, figures_dir)
    ablation = evaluate_feature_ablation(feature_tables.by_case_wide, reports_dir, figures_dir)
    sensor = evaluate_sensor_count(feature_tables.by_bus, reports_dir, figures_dir)
    localization = evaluate_fault_localization(feature_tables.by_case_wide, Path("data/ieee30_edges.csv"), reports_dir, figures_dir)

    final_summary = build_final_summary(case_index, quality_df, feature_tables, full_cls.metrics, full_cls.predictions, ablation.metrics, sensor.curve, sensor.selected, localization)
    (reports_dir / "final_analysis_summary.md").write_text(final_summary, encoding="utf-8")

    write_repo_results(reports_dir, figures_dir)
    print(f"Processed {len(cases)} waveform cases")
    if archive_dir is not None:
        print(f"Archived previous outputs to: {archive_dir}")
    print(f"Reports: {reports_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
