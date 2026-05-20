from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wmu_project.waveform_classification import evaluate_feature_ablation, evaluate_full_classification
from wmu_project.waveform_features import FeatureTables, compute_feature_tables, export_feature_tables
from wmu_project.waveform_io import list_cases, output_roots
from wmu_project.waveform_localization import evaluate_fault_localization
from wmu_project.waveform_quality import build_quality_outputs, write_quality_outputs
from wmu_project.waveform_sensor_selection import evaluate_sensor_count
from wmu_project.waveform_utils import build_case_index_frame, to_local_path


REPO_RESULTS = Path("results/waveform_event_analysis")


def load_existing_feature_tables(output_dir: Path) -> FeatureTables | None:
    bus_path = output_dir / "feature_table_by_bus.csv"
    case_path = output_dir / "feature_table_by_case.csv"
    wide_path = output_dir / "feature_table_by_case_wide.csv"
    if bus_path.exists() and case_path.exists() and wide_path.exists():
        return FeatureTables(
            by_bus=pd.read_csv(bus_path),
            by_case=pd.read_csv(case_path),
            by_case_wide=pd.read_csv(wide_path),
        )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WMU waveform event analysis pipeline")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--f0", type=float, default=50.0)
    return parser.parse_args()


def write_repo_results(src_reports: Path, src_figures: Path) -> None:
    repo_reports = REPO_RESULTS / "reports"
    repo_figures = REPO_RESULTS / "figures"
    repo_reports.mkdir(parents=True, exist_ok=True)
    repo_figures.mkdir(parents=True, exist_ok=True)
    for name in [
        "data_quality_summary.md",
        "classification_full_wmu_metrics.csv",
        "classification_full_wmu_report.txt",
        "feature_ablation_metrics.csv",
        "sensor_count_curve.csv",
        "selected_wmu_by_k.csv",
        "fault_localization_preliminary.csv",
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
    tied_ablation = ablation_metrics.loc[ablation_metrics["macro_f1"] == best_ablation["macro_f1"], ["FeatureGroup", "Model"]].drop_duplicates()
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
        f"- Input xlsx total: {len(cases_df)}",
        f"- Class counts: {class_counts}",
        f"- Quality status counts: OK={status_counts.get('OK', 0)}, WARNING={status_counts.get('WARNING', 0)}, FAILED={status_counts.get('FAILED', 0)}",
        f"- Generated bus-level feature count: {len([c for c in feature_tables.by_bus.columns if c not in {'CaseName','EventType','TargetBus','ObservedBus','EventTime','SamplingRateHz'}])}",
        f"- Full-WMU best model: {best_cls['Model']} | macro-F1={best_cls['macro_f1']:.4f} | balanced_accuracy={best_cls['balanced_accuracy']:.4f}",
        f"- Best feature ablation group(s): {", ".join(sorted(tied_ablation["FeatureGroup"].unique()))} | top macro-F1={best_ablation['macro_f1']:.4f}",
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
        "## Research interpretation",
        "- DV_energy remains a strong event-trigger feature, but it is not sufficient alone for robust four-class separation when LoadSwitch must be separated from faults.",
        "- Sag-only behavior can highlight disturbance severity, but current/unbalance/sequence-style features are needed to differentiate balanced vs unbalanced faults and switching events.",
        "- Time-frequency and current/sequence features materially improve LoadSwitch/Fault separation when waveform shapes share comparable voltage dips.",
        "- WMU placement should maximize discriminative classification value under limited sensor budgets, not just fault-trigger energy or binary detection coverage.",
        "",
        "## Next steps",
        "- Expand LoadSwitch intensity levels to 5%, 15%, and 25%.",
        "- Vary fault resistance, clearing time, and inception angle.",
        "- Build a larger dataset with train/test separation across repeated conditions.",
        "- Extend to weak-grid or DER-integrated operating scenarios.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_dir = to_local_path(args.input_dir)
    output_dir = to_local_path(args.output_dir)
    reports_dir, figures_dir = output_roots(output_dir)

    cases = list_cases(input_dir)
    case_index = build_case_index_frame(cases)
    case_index.to_csv(reports_dir / "dataset_case_index.csv", index=False)

    quality_report_path = reports_dir / "data_quality_report.csv"
    if quality_report_path.exists() and (reports_dir / "data_quality_summary.txt").exists():
        quality_df = pd.read_csv(quality_report_path)
        quality_summary = (reports_dir / "data_quality_summary.txt").read_text(encoding="utf-8")
    else:
        quality_df, quality_summary = build_quality_outputs(cases)
        write_quality_outputs(quality_df, quality_summary, reports_dir, reports_dir / "data_quality_summary.md")

    feature_tables = load_existing_feature_tables(output_dir)
    if feature_tables is None:
        feature_tables = compute_feature_tables(cases, f0=args.f0)
        export_feature_tables(feature_tables, output_dir)

    full_cls = evaluate_full_classification(feature_tables.by_case_wide, reports_dir, figures_dir)
    ablation = evaluate_feature_ablation(feature_tables.by_case_wide, reports_dir, figures_dir)
    sensor = evaluate_sensor_count(feature_tables.by_case_wide, reports_dir, figures_dir)
    localization = evaluate_fault_localization(feature_tables.by_case_wide, Path("data/ieee30_edges.csv"), reports_dir, figures_dir)

    final_summary = build_final_summary(case_index, quality_df, feature_tables, full_cls.metrics, full_cls.predictions, ablation.metrics, sensor.curve, sensor.selected, localization)
    (reports_dir / "final_analysis_summary.md").write_text(final_summary, encoding="utf-8")

    write_repo_results(reports_dir, figures_dir)
    print(f"Processed {len(cases)} waveform cases")
    print(f"Reports: {reports_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
