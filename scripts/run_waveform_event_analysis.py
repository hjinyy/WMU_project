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

REPO_RESULTS_DEFAULT = Path('results/waveform_event_analysis')
REPO_RESULTS_IBR_BACKGROUND = Path('results/waveform_ibr_background_analysis')
EXPECTED_COUNTS_BASELINE = {
    'Normal': 3,
    'LoadSwitch': 20,
    'SLG_Fault': 30,
    'ThreePhase_Fault': 30,
}
EXPECTED_COUNTS_IBR_BACKGROUND = {
    'SSO_Normal': 3,
    'SSO_LoadSwitch': 21,
    'SSO_SLG_Fault': 30,
    'SSO_ThreePhase_Fault': 30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run WMU waveform event analysis pipeline')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--f0', type=float, default=50.0)
    parser.add_argument(
        '--scenario',
        choices=['baseline', 'ibr-background', 'auto'],
        default='auto',
        help='Expected inventory and tracked repo-results namespace.',
    )
    return parser.parse_args()


def archive_existing_outputs(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / 'reports',
        output_dir / 'figures',
        output_dir / 'feature_table_by_bus.csv',
        output_dir / 'feature_table_by_case.csv',
        output_dir / 'feature_table_by_case_wide.csv',
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    archive_root = output_dir / 'archive'
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    archive_dir = archive_root / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), archive_dir / path.name)
    return archive_dir


def resolve_scenario(cases_df: pd.DataFrame, requested: str) -> str:
    if requested != 'auto':
        return requested
    events = set(cases_df['EventType'].astype(str))
    return 'ibr-background' if any(event.startswith('SSO_') for event in events) else 'baseline'


def expected_counts_for_scenario(scenario: str) -> dict[str, int]:
    return EXPECTED_COUNTS_IBR_BACKGROUND if scenario == 'ibr-background' else EXPECTED_COUNTS_BASELINE


def repo_results_for_scenario(scenario: str) -> Path:
    return REPO_RESULTS_IBR_BACKGROUND if scenario == 'ibr-background' else REPO_RESULTS_DEFAULT


def verify_input_inventory(cases_df: pd.DataFrame, scenario: str) -> None:
    expected_counts = expected_counts_for_scenario(scenario)
    expected_total = sum(expected_counts.values())
    counts = cases_df['EventType'].astype(str).value_counts().to_dict()
    total = len(cases_df)
    if total != expected_total or any(counts.get(key, 0) != expected_counts[key] for key in expected_counts):
        raise RuntimeError(f'Input inventory mismatch: total={total}, counts={counts}, expected_total={expected_total}, expected_counts={expected_counts}')


def write_repo_results(src_reports: Path, src_figures: Path, repo_results: Path) -> None:
    repo_reports = repo_results / 'reports'
    repo_figures = repo_results / 'figures'
    repo_reports.mkdir(parents=True, exist_ok=True)
    repo_figures.mkdir(parents=True, exist_ok=True)
    report_names = [
        'data_quality_summary.md',
        'classification_full_wmu_metrics.csv',
        'classification_flat_vs_hierarchical_metrics.csv',
        'classification_full_wmu_report.txt',
        'confusion_matrix_full_wmu.csv',
        'confusion_matrix_hierarchical_full_wmu.csv',
        'misclassified_cases_full_wmu.csv',
        'misclassified_cases_hierarchical_full_wmu.csv',
        'hierarchical_trigger_debug.csv',
        'feature_ablation_metrics.csv',
        'feature_ablation_used_columns.txt',
        'feature_ablation_trigger_debug.csv',
        'sensor_count_curve.csv',
        'selected_wmu_by_k.csv',
        'sensor_selection_debug.csv',
        'fault_localization_preliminary.csv',
        'fault_localization_debug.csv',
        'fault_localization_notes.txt',
        'final_analysis_summary.md',
        'dataset_case_index.csv',
    ]
    figure_names = [
        'confusion_matrix_full_wmu.png',
        'confusion_matrix_hierarchical_full_wmu.png',
        'feature_ablation_macro_f1.png',
        'feature_ablation_normal_recall.png',
        'feature_ablation_loadswitch_recall.png',
        'sensor_count_macro_f1.png',
        'sensor_count_balanced_accuracy.png',
        'sensor_count_normal_recall.png',
        'sensor_count_loadswitch_recall.png',
        'sensor_count_fault_precision.png',
        'fault_localization_confusion_matrix.png',
    ]
    for name in report_names:
        src = src_reports / name
        if src.exists():
            (repo_reports / name).write_bytes(src.read_bytes())
    for name in figure_names:
        src = src_figures / name
        if src.exists():
            (repo_figures / name).write_bytes(src.read_bytes())


def build_final_summary(cases_df: pd.DataFrame, quality_df: pd.DataFrame, feature_tables, cls_metrics: pd.DataFrame, cls_predictions: pd.DataFrame, ablation_metrics: pd.DataFrame, sensor_curve: pd.DataFrame, selected_wmu: pd.DataFrame, localization_summary: pd.DataFrame, scenario: str) -> str:
    status_counts = quality_df['Status'].value_counts().to_dict()
    class_counts = cases_df['EventType'].astype(str).value_counts().to_dict()
    flat_metrics = cls_metrics.loc[cls_metrics['Strategy'] == 'flat'].sort_values(['macro_f1', 'balanced_accuracy'], ascending=False)
    hier_metrics = cls_metrics.loc[cls_metrics['Strategy'] == 'hierarchical'].sort_values(['macro_f1', 'balanced_accuracy', 'Normal_recall'], ascending=False)
    best_flat = flat_metrics.iloc[0]
    best_hier = hier_metrics.iloc[0]
    best_ablation = ablation_metrics.sort_values(['Strategy', 'FeatureGroup', 'macro_f1', 'Normal_recall'], ascending=[True, True, False, False]).groupby(['Strategy', 'FeatureGroup'], as_index=False).first()
    greedy_curve = sensor_curve.loc[sensor_curve['Method'] == 'feature_aware_greedy'].sort_values('k')
    max_f1 = float(greedy_curve['macro_f1'].max())
    plateau_threshold = max_f1 - 0.01
    plateau_row = greedy_curve.loc[greedy_curve['macro_f1'] >= plateau_threshold].sort_values('k').iloc[0]
    best_k = int(plateau_row['k'])
    selected_k = selected_wmu.loc[(selected_wmu['Method'] == 'feature_aware_greedy') & (selected_wmu['k'] <= best_k), 'SelectedBus'].tolist()

    best_flat_mask = (
        (cls_predictions['Strategy'] == best_flat['Strategy'])
        & (cls_predictions['EventModel'] == best_flat['EventModel'])
        & (cls_predictions['TriggerMethod'] == best_flat['TriggerMethod'])
        & (cls_predictions['TriggerModel'] == best_flat['TriggerModel'])
    )
    best_hier_mask = (
        (cls_predictions['Strategy'] == best_hier['Strategy'])
        & (cls_predictions['EventModel'] == best_hier['EventModel'])
        & (cls_predictions['TriggerMethod'] == best_hier['TriggerMethod'])
        & (cls_predictions['TriggerModel'] == best_hier['TriggerModel'])
    )
    ls_fault_cases = cls_predictions.loc[best_hier_mask & cls_predictions['LoadSwitchToFault'], ['CaseName', 'PredictedEventType']].drop_duplicates()

    lines = [
        '# Final Waveform Event Analysis Summary',
        '',
        f'- Scenario: {scenario}',
        f'- Input raw event files: {len(cases_df)}',
        f'- Class counts: {class_counts}',
    ]
    if scenario == 'ibr-background':
        lines.extend([
            '- All cases include the IBR-like 25 Hz SSO background condition.',
            '- SSO_LoadSwitch cases use 21 target buses with LoadAdd P/QL set to 15% of the existing Load bus P/QL.',
            '- LoadSwitch event time is 0.1 s; fault start/clear times are 0.3/0.36 s.',
        ])
    else:
        lines.append('- LoadSwitch case count is 20 because only buses with existing Pd or Qd were regenerated under the 15% abrupt load-increase condition.')
    lines.extend([
        f"- Quality status counts: OK={status_counts.get('OK', 0)}, WARNING={status_counts.get('WARNING', 0)}, FAILED={status_counts.get('FAILED', 0)}",
        f"- Generated bus-level feature count: {len([c for c in feature_tables.by_bus.columns if c not in {'CaseName','EventType','TargetBus','ObservedBus','EventTime','SamplingRateHz'}])}",
        f"- Best flat classifier: {best_flat['EventModel']} | macro-F1={best_flat['macro_f1']:.4f} | balanced_accuracy={best_flat['balanced_accuracy']:.4f} | Normal_recall={best_flat['Normal_recall']:.4f}",
        f"- Best hierarchical classifier: {best_hier['TriggerMethod']} + {best_hier['EventModel']} | macro-F1={best_hier['macro_f1']:.4f} | balanced_accuracy={best_hier['balanced_accuracy']:.4f} | Normal_recall={best_hier['Normal_recall']:.4f}",
        f'- Sensor-count saturation k (within 0.01 macro-F1 of best greedy result): {best_k}',
        f'- Selected WMU combination up to saturation k: {selected_k}',
        '',
        '## Best ablation rows',
        '```',
        best_ablation[['FeatureGroup', 'Strategy', 'TriggerMethod', 'EventModel', 'macro_f1', 'Normal_recall', 'LoadSwitch_recall', 'Fault_precision']].to_string(index=False),
        '```',
        '',
        '## LoadSwitch misclassified as Fault (best hierarchical)',
    ])
    if ls_fault_cases.empty:
        lines.append('- none')
    else:
        for _, row in ls_fault_cases.iterrows():
            lines.append(f"- {row['CaseName']} -> {row['PredictedEventType']}")
    lines.extend([
        '',
        '## Fault localization preliminary',
        '```',
        localization_summary.to_string(index=False),
        '```',
        '',
        '## Current limitations',
    ])
    if scenario == 'ibr-background':
        lines.extend([
            '- The IBR-like SSO background is intentionally present in every class, so event classification measures the incremental disturbance on top of the shared oscillatory background.',
            '- The SSO frequency and envelope are fixed at one configured condition.',
        ])
    lines.extend([
        '- Normal class has only 3 cases.',
        '- LoadSwitch cases currently cover only one disturbance strength: 15% load increase.',
        '- Fault resistance, clearing time, and inception angle are not varied.',
        '- Fault localization remains preliminary because there is only one case per bus per fault type.',
        '- If raw event files are mislabeled or duplicated across classes, classifier metrics should be treated as data-integrity diagnostics rather than final performance claims.',
        '',
        '## Next steps',
        '- Audit SLG raw files against the Normal baseline when event-trigger features remain near zero.',
        '- Expand LoadSwitch intensity levels to 5%, 15%, and 25%.',
        '- Vary fault resistance.',
        '- Vary clearing time and inception angle.',
        '- Build a larger dataset with train/test separation across repeated conditions.',
        '- Extend to weak-grid or DER-integrated operating scenarios.',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    input_dir = to_local_path(args.input_dir)
    output_dir = to_local_path(args.output_dir)

    print('Listing waveform cases...', flush=True)
    cases = list_cases(input_dir)
    case_index = build_case_index_frame(cases)
    scenario = resolve_scenario(case_index, args.scenario)
    verify_input_inventory(case_index, scenario)
    repo_results = repo_results_for_scenario(scenario)

    archive_dir = archive_existing_outputs(output_dir)
    reports_dir, figures_dir = output_roots(output_dir)
    case_index.to_csv(reports_dir / 'dataset_case_index.csv', index=False)

    print('Running data quality checks...', flush=True)
    quality_df, quality_summary = build_quality_outputs(cases, max_workers=1)
    write_quality_outputs(quality_df, quality_summary, reports_dir, reports_dir / 'data_quality_summary.md')

    print('Computing feature tables...', flush=True)
    feature_tables = compute_feature_tables(cases, f0=args.f0, max_workers=1)
    export_feature_tables(feature_tables, output_dir)

    print('Evaluating full-WMU classification...', flush=True)
    full_cls = evaluate_full_classification(feature_tables.by_case_wide, reports_dir, figures_dir)
    print('Evaluating feature ablation...', flush=True)
    ablation = evaluate_feature_ablation(feature_tables.by_case_wide, reports_dir, figures_dir)
    print('Evaluating sensor count curve...', flush=True)
    sensor = evaluate_sensor_count(feature_tables.by_bus, reports_dir, figures_dir, random_trials=3)
    print('Evaluating fault localization...', flush=True)
    localization = evaluate_fault_localization(feature_tables.by_case_wide, Path('data/ieee30_edges.csv'), reports_dir, figures_dir)

    final_summary = build_final_summary(case_index, quality_df, feature_tables, full_cls.metrics, full_cls.predictions, ablation.metrics, sensor.curve, sensor.selected, localization, scenario)
    (reports_dir / 'final_analysis_summary.md').write_text(final_summary, encoding='utf-8')

    write_repo_results(reports_dir, figures_dir, repo_results)
    print(f'Processed {len(cases)} waveform cases for scenario={scenario}')
    if archive_dir is not None:
        print(f'Archived previous outputs to: {archive_dir}')
    print(f'Reports: {reports_dir}')
    print(f'Figures: {figures_dir}')


if __name__ == '__main__':
    main()
