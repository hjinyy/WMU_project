from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

from .waveform_features import select_feature_columns_by_group
from .waveform_utils import EVENT_CLASS_ORDER, FAULT_EVENTS

RANDOM_SEED = 42
EVENT_LABELS = [label for label in EVENT_CLASS_ORDER if label != 'Normal']
TRIGGER_CANDIDATES = ('max_dV_energy', 'max_dI_energy', 'max_sag', 'max_I_rms_jump', 'max_Z_drop_ratio')


@dataclass
class EvaluationArtifacts:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    stratified_metrics: pd.DataFrame
    feature_columns: dict[str, list[str]] | None = None
    trigger_details: pd.DataFrame | None = None


def build_models() -> dict[str, Pipeline]:
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    tree_numeric = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    return {
        'LogisticRegression': Pipeline([
            ('prep', numeric),
            ('model', OneVsRestClassifier(LogisticRegression(max_iter=5000, class_weight='balanced', solver='liblinear', random_state=RANDOM_SEED))),
        ]),
        'RandomForest': Pipeline([
            ('prep', tree_numeric),
            ('model', RandomForestClassifier(n_estimators=400, random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1)),
        ]),
        'LinearSVC': Pipeline([
            ('prep', numeric),
            ('model', LinearSVC(class_weight='balanced', random_state=RANDOM_SEED, dual='auto', max_iter=5000)),
        ]),
    }


def build_binary_models() -> dict[str, Pipeline]:
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    tree_numeric = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    return {
        'LogisticRegression': Pipeline([
            ('prep', numeric),
            ('model', LogisticRegression(max_iter=5000, class_weight='balanced', solver='liblinear', random_state=RANDOM_SEED)),
        ]),
        'RandomForest': Pipeline([
            ('prep', tree_numeric),
            ('model', RandomForestClassifier(n_estimators=400, random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1)),
        ]),
        'LinearSVC': Pipeline([
            ('prep', numeric),
            ('model', LinearSVC(class_weight='balanced', random_state=RANDOM_SEED, dual='auto', max_iter=5000)),
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


def compute_metrics(truth: np.ndarray, pred: np.ndarray, labels: list[str], model_name: str, feature_group: str, validation: str, strategy: str = 'flat', trigger_method: str = 'none', trigger_model: str = 'none', notes: str = '') -> dict[str, object]:
    per_class = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
    precision, recall, f1, support = per_class
    row: dict[str, object] = {
        'FeatureGroup': feature_group,
        'Strategy': strategy,
        'TriggerMethod': trigger_method,
        'TriggerModel': trigger_model,
        'EventModel': model_name,
        'Model': model_name,
        'Validation': validation,
        'accuracy': accuracy_score(truth, pred),
        'macro_f1': f1_score(truth, pred, labels=labels, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(truth, pred),
        'Notes': notes,
    }
    for idx, label in enumerate(labels):
        row[f'precision_{label}'] = precision[idx]
        row[f'recall_{label}'] = recall[idx]
        row[f'f1_{label}'] = f1[idx]
        row[f'support_{label}'] = support[idx]
    truth_fault = np.isin(truth, list(FAULT_EVENTS))
    pred_fault = np.isin(pred, list(FAULT_EVENTS))
    tp = int(np.sum(truth_fault & pred_fault))
    fp = int(np.sum(~truth_fault & pred_fault))
    row['Fault_precision'] = tp / (tp + fp) if (tp + fp) else 0.0
    normal_mask = truth == 'Normal'
    row['Normal_false_alarm_rate'] = float(np.mean(pred[normal_mask] != 'Normal')) if normal_mask.any() else np.nan
    row['Normal_recall'] = row.get('recall_Normal', np.nan)
    row['LoadSwitch_recall'] = row.get('recall_LoadSwitch', np.nan)
    row['LoadSwitch_to_Fault_count'] = int(np.sum((truth == 'LoadSwitch') & np.isin(pred, list(FAULT_EVENTS))))
    return row


def aggregate_trigger_features(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    allowed = set(feature_columns or frame.columns.tolist())
    data: dict[str, pd.Series] = {}

    def aggregate_case_or_bus(target_name: str, case_col: str | None, bus_suffix: str | None) -> None:
        if case_col and case_col in frame.columns and case_col in allowed:
            data[target_name] = pd.to_numeric(frame[case_col], errors='coerce')
            return
        if not bus_suffix:
            data[target_name] = pd.Series(np.nan, index=frame.index)
            return
        cols = [col for col in frame.columns if col in allowed and col.startswith('Bus') and col.endswith(f'__{bus_suffix}')]
        if cols:
            data[target_name] = frame[cols].apply(pd.to_numeric, errors='coerce').max(axis=1)
        else:
            data[target_name] = pd.Series(np.nan, index=frame.index)

    aggregate_case_or_bus('max_dV_energy', 'max_dV_energy', 'dV_energy_3ph_max')
    aggregate_case_or_bus('max_dI_energy', 'max_dI_energy', 'dI_energy_3ph_max')
    aggregate_case_or_bus('max_sag', 'max_sag', 'max_sag')
    aggregate_case_or_bus('max_Z_drop_ratio', 'max_Z_drop_ratio', 'Z_drop_ratio')
    aggregate_case_or_bus('max_I_rms_jump', None, 'I_rms_jump_3ph_max')

    trigger = pd.DataFrame(data)
    trigger = trigger.replace([np.inf, -np.inf], np.nan)
    return trigger


def build_rule_thresholds(train_trigger: pd.DataFrame, train_labels: pd.Series, margin: float = 1.10) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    normal_rows = train_trigger.loc[train_labels == 'Normal']
    for feature in train_trigger.columns:
        series = pd.to_numeric(normal_rows[feature], errors='coerce').dropna()
        if series.empty:
            thresholds[feature] = np.inf
            continue
        base = float(series.max())
        eps = max(1e-12, abs(base) * 1e-3)
        thresholds[feature] = base * margin + eps
    return thresholds


def apply_rule_trigger(row: pd.Series, thresholds: dict[str, float]) -> tuple[bool, float, str]:
    normalized: dict[str, float] = {}
    for feature, threshold in thresholds.items():
        value = float(pd.to_numeric(pd.Series([row.get(feature)]), errors='coerce').iloc[0]) if pd.notna(row.get(feature)) else 0.0
        if not np.isfinite(value):
            value = 0.0
        if not np.isfinite(threshold) or threshold <= 0:
            normalized[feature] = 0.0
        else:
            normalized[feature] = value / threshold
    top_feature = max(normalized, key=normalized.get) if normalized else 'none'
    score = float(max(normalized.values())) if normalized else 0.0
    return score > 1.0, score, top_feature


def hierarchical_loo_predict(by_case_wide: pd.DataFrame, feature_columns: list[str], event_model_name: str, trigger_method: str, trigger_model_name: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    event_models = build_models()
    binary_models = build_binary_models()
    event_model = event_models[event_model_name]

    feature_x = by_case_wide[feature_columns].copy()
    labels = by_case_wide['EventType'].astype(str).reset_index(drop=True)
    trigger_x = aggregate_trigger_features(by_case_wide, feature_columns).reset_index(drop=True)

    loo = LeaveOneOut()
    preds = np.empty(labels.shape[0], dtype=object)
    trigger_rows: list[dict[str, object]] = []

    for train_idx, test_idx in loo.split(feature_x, labels):
        test_i = int(test_idx[0])
        x_train = feature_x.iloc[train_idx]
        x_test = feature_x.iloc[test_idx]
        y_train = labels.iloc[train_idx]
        train_trigger = trigger_x.iloc[train_idx]
        test_trigger = trigger_x.iloc[test_idx[0]]

        if trigger_method == 'rule_based':
            thresholds = build_rule_thresholds(train_trigger, y_train)
            is_event, trigger_score, trigger_feature = apply_rule_trigger(test_trigger, thresholds)
            trigger_model = 'RuleTrigger'
            threshold_note = '; '.join(f"{key}={value:.4g}" for key, value in thresholds.items())
        elif trigger_method == 'binary_ml':
            trigger_model = trigger_model_name or 'LogisticRegression'
            binary_model = clone(binary_models[trigger_model])
            y_train_binary = np.where(y_train == 'Normal', 'Normal', 'Event')
            binary_model.fit(train_trigger, y_train_binary)
            pred_binary = binary_model.predict(trigger_x.iloc[test_idx])[0]
            is_event = pred_binary == 'Event'
            trigger_score = float('nan')
            trigger_feature = 'ml_trigger'
            threshold_note = ''
        else:
            raise KeyError(f'Unknown trigger method: {trigger_method}')

        if not is_event:
            preds[test_i] = 'Normal'
            trigger_rows.append({
                'CaseName': by_case_wide.iloc[test_i]['CaseName'],
                'TrueEventType': labels.iloc[test_i],
                'PredictedEventType': 'Normal',
                'TriggerMethod': trigger_method,
                'TriggerModel': trigger_model,
                'EventModel': event_model_name,
                'TriggerDecision': 'Normal',
                'TriggerScore': trigger_score,
                'TriggerTopFeature': trigger_feature,
                'TriggerThresholds': threshold_note,
            })
            continue

        event_mask = y_train != 'Normal'
        x_train_event = x_train.loc[event_mask]
        y_train_event = y_train.loc[event_mask]
        fitted_event = clone(event_model)
        fitted_event.fit(x_train_event, y_train_event)
        pred_event = fitted_event.predict(x_test)[0]
        preds[test_i] = pred_event
        trigger_rows.append({
            'CaseName': by_case_wide.iloc[test_i]['CaseName'],
            'TrueEventType': labels.iloc[test_i],
            'PredictedEventType': pred_event,
            'TriggerMethod': trigger_method,
            'TriggerModel': trigger_model,
            'EventModel': event_model_name,
            'TriggerDecision': 'Event',
            'TriggerScore': trigger_score,
            'TriggerTopFeature': trigger_feature,
            'TriggerThresholds': threshold_note,
        })

    return preds.astype(str), pd.DataFrame(trigger_rows)


def evaluate_models_for_group(by_case_wide: pd.DataFrame, feature_group: str, models: dict[str, Pipeline] | None = None, selected_models: list[str] | None = None) -> EvaluationArtifacts:
    models = models or build_models()
    labels = [label for label in EVENT_CLASS_ORDER if label in by_case_wide['EventType'].unique()]
    feature_columns = select_feature_columns_by_group(by_case_wide, feature_group)
    x = by_case_wide[feature_columns].copy()
    y = by_case_wide['EventType'].astype(str).to_numpy()
    metrics_rows = []
    pred_rows = []
    strat_rows = []
    active_models = selected_models or list(models)
    for model_name in active_models:
        model = models[model_name]
        pred = loo_predict(model, x, y)
        metrics_rows.append(compute_metrics(y, pred, labels, model_name, feature_group, 'LOO', strategy='flat'))
        pred_df = by_case_wide[['CaseName', 'EventType', 'TargetBus']].copy()
        pred_df['Strategy'] = 'flat'
        pred_df['TriggerMethod'] = 'none'
        pred_df['TriggerModel'] = 'none'
        pred_df['EventModel'] = model_name
        pred_df['Model'] = model_name
        pred_df['FeatureGroup'] = feature_group
        pred_df['PredictedEventType'] = pred
        pred_df['IsCorrect'] = pred_df['EventType'] == pred_df['PredictedEventType']
        pred_df['LoadSwitchToFault'] = (pred_df['EventType'] == 'LoadSwitch') & pred_df['PredictedEventType'].isin(FAULT_EVENTS)
        pred_rows.append(pred_df)


    return EvaluationArtifacts(
        metrics=pd.DataFrame(metrics_rows).sort_values(['macro_f1', 'balanced_accuracy'], ascending=False).reset_index(drop=True),
        predictions=pd.concat(pred_rows, ignore_index=True),
        stratified_metrics=pd.DataFrame(strat_rows),
        feature_columns={feature_group: feature_columns},
    )


def evaluate_hierarchical_for_group(by_case_wide: pd.DataFrame, feature_group: str, event_models: list[str] | None = None, binary_trigger_models: list[str] | None = None) -> EvaluationArtifacts:
    labels = [label for label in EVENT_CLASS_ORDER if label in by_case_wide['EventType'].unique()]
    feature_columns = select_feature_columns_by_group(by_case_wide, feature_group)
    y = by_case_wide['EventType'].astype(str).to_numpy()
    metrics_rows = []
    pred_rows = []
    trigger_rows = []
    event_models = event_models or ['RandomForest']
    binary_trigger_models = binary_trigger_models or ['LogisticRegression', 'RandomForest', 'LinearSVC']

    for event_model in event_models:
        pred, trigger_df = hierarchical_loo_predict(by_case_wide, feature_columns, event_model_name=event_model, trigger_method='rule_based')
        metrics_rows.append(compute_metrics(y, pred, labels, event_model, feature_group, 'LOO', strategy='hierarchical', trigger_method='rule_based', trigger_model='RuleTrigger'))
        pred_df = by_case_wide[['CaseName', 'EventType', 'TargetBus']].copy()
        pred_df['Strategy'] = 'hierarchical'
        pred_df['TriggerMethod'] = 'rule_based'
        pred_df['TriggerModel'] = 'RuleTrigger'
        pred_df['EventModel'] = event_model
        pred_df['Model'] = event_model
        pred_df['FeatureGroup'] = feature_group
        pred_df['PredictedEventType'] = pred
        pred_df['IsCorrect'] = pred_df['EventType'] == pred_df['PredictedEventType']
        pred_df['LoadSwitchToFault'] = (pred_df['EventType'] == 'LoadSwitch') & pred_df['PredictedEventType'].isin(FAULT_EVENTS)
        pred_rows.append(pred_df)
        trigger_rows.append(trigger_df.assign(FeatureGroup=feature_group, Strategy='hierarchical'))

    for trigger_model in binary_trigger_models:
        for event_model in event_models:
            pred, trigger_df = hierarchical_loo_predict(by_case_wide, feature_columns, event_model_name=event_model, trigger_method='binary_ml', trigger_model_name=trigger_model)
            metrics_rows.append(compute_metrics(y, pred, labels, event_model, feature_group, 'LOO', strategy='hierarchical', trigger_method='binary_ml', trigger_model=trigger_model))
            pred_df = by_case_wide[['CaseName', 'EventType', 'TargetBus']].copy()
            pred_df['Strategy'] = 'hierarchical'
            pred_df['TriggerMethod'] = 'binary_ml'
            pred_df['TriggerModel'] = trigger_model
            pred_df['EventModel'] = event_model
            pred_df['Model'] = f'{trigger_model}->{event_model}'
            pred_df['FeatureGroup'] = feature_group
            pred_df['PredictedEventType'] = pred
            pred_df['IsCorrect'] = pred_df['EventType'] == pred_df['PredictedEventType']
            pred_df['LoadSwitchToFault'] = (pred_df['EventType'] == 'LoadSwitch') & pred_df['PredictedEventType'].isin(FAULT_EVENTS)
            pred_rows.append(pred_df)
            trigger_rows.append(trigger_df.assign(FeatureGroup=feature_group, Strategy='hierarchical'))

    metrics = pd.DataFrame(metrics_rows).sort_values(['macro_f1', 'balanced_accuracy', 'Normal_recall'], ascending=[False, False, False]).reset_index(drop=True)
    predictions = pd.concat(pred_rows, ignore_index=True)
    trigger_details = pd.concat(trigger_rows, ignore_index=True)
    return EvaluationArtifacts(metrics=metrics, predictions=predictions, stratified_metrics=pd.DataFrame(), feature_columns={feature_group: feature_columns}, trigger_details=trigger_details)


def write_confusion_matrix_csv(truth: np.ndarray, pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    cm = confusion_matrix(truth, pred, labels=labels)
    frame = pd.DataFrame(cm, index=labels, columns=labels)
    frame.index.name = 'True'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)


def plot_confusion_matrix(truth: np.ndarray, pred: np.ndarray, labels: list[str], title: str, output_path: Path) -> None:
    cm = confusion_matrix(truth, pred, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _describe_best_result(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty:
        return ['- no result']
    best = metrics.iloc[0]
    return [
        f"- Strategy={best['Strategy']} | TriggerMethod={best['TriggerMethod']} | TriggerModel={best['TriggerModel']} | EventModel={best['EventModel']}",
        f"- macro-F1={best['macro_f1']:.4f} | balanced_accuracy={best['balanced_accuracy']:.4f}",
        f"- Normal_recall={best.get('Normal_recall', np.nan):.4f} | LoadSwitch_recall={best['LoadSwitch_recall']:.4f} | Fault_precision={best['Fault_precision']:.4f}",
    ]


def write_classification_report(flat_metrics: pd.DataFrame, flat_predictions: pd.DataFrame, hierarchical_metrics: pd.DataFrame, hierarchical_predictions: pd.DataFrame, output_path: Path) -> None:
    lines = [
        'Waveform event classification report',
        '',
        '## Best flat 4-class result',
        *_describe_best_result(flat_metrics),
        '',
        '## Best hierarchical result',
        *_describe_best_result(hierarchical_metrics),
        '',
        '## Caveats',
        '- Normal class has only 3 samples, so Normal recall is highly unstable.',
        '- Hierarchical triggering is intended to separate no-event vs event first; it does not repair mislabeled or corrupted raw event files.',
        '',
        '## LoadSwitch -> Fault misclassifications (best flat)',
    ]
    if not flat_metrics.empty:
        best_flat = flat_metrics.iloc[0]
        flat_mask = (
            (flat_predictions['Strategy'] == best_flat['Strategy'])
            & (flat_predictions['FeatureGroup'] == best_flat['FeatureGroup'])
            & (flat_predictions['EventModel'] == best_flat['EventModel'])
            & (flat_predictions['TriggerMethod'] == best_flat['TriggerMethod'])
            & (flat_predictions['TriggerModel'] == best_flat['TriggerModel'])
        )
        fault_cases = flat_predictions.loc[flat_mask & flat_predictions['LoadSwitchToFault'], ['CaseName', 'PredictedEventType']]
        if fault_cases.empty:
            lines.append('- none')
        else:
            for _, row in fault_cases.iterrows():
                lines.append(f"- {row['CaseName']} -> {row['PredictedEventType']}")
    else:
        lines.append('- none')

    lines.extend(['', '## Metrics tables', '', '[Flat]', flat_metrics.to_string(index=False), '', '[Hierarchical]', hierarchical_metrics.to_string(index=False)])
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _best_prediction_subset(predictions: pd.DataFrame, metrics_row: pd.Series) -> pd.DataFrame:
    return predictions.loc[
        (predictions['Strategy'] == metrics_row['Strategy'])
        & (predictions['FeatureGroup'] == metrics_row['FeatureGroup'])
        & (predictions['EventModel'] == metrics_row['EventModel'])
        & (predictions['TriggerMethod'] == metrics_row['TriggerMethod'])
        & (predictions['TriggerModel'] == metrics_row['TriggerModel'])
    ].copy()


def evaluate_full_classification(by_case_wide: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> EvaluationArtifacts:
    flat = evaluate_models_for_group(by_case_wide, 'All_features', selected_models=['LogisticRegression', 'RandomForest'])
    hierarchical = evaluate_hierarchical_for_group(by_case_wide, 'All_features', event_models=['RandomForest'], binary_trigger_models=['RandomForest'])

    labels = [label for label in EVENT_CLASS_ORDER if label in by_case_wide['EventType'].unique()]
    flat.metrics.to_csv(reports_dir / 'classification_full_wmu_metrics.csv', index=False)
    combined = pd.concat([flat.metrics, hierarchical.metrics], ignore_index=True).sort_values(['Strategy', 'macro_f1', 'balanced_accuracy'], ascending=[True, False, False]).reset_index(drop=True)
    combined.to_csv(reports_dir / 'classification_flat_vs_hierarchical_metrics.csv', index=False)
    if hierarchical.trigger_details is not None:
        hierarchical.trigger_details.to_csv(reports_dir / 'hierarchical_trigger_debug.csv', index=False)
    write_classification_report(flat.metrics, flat.predictions, hierarchical.metrics, hierarchical.predictions, reports_dir / 'classification_full_wmu_report.txt')

    best_flat = flat.metrics.iloc[0]
    best_flat_pred = _best_prediction_subset(flat.predictions, best_flat)
    write_confusion_matrix_csv(best_flat_pred['EventType'].to_numpy(), best_flat_pred['PredictedEventType'].to_numpy(), labels, reports_dir / 'confusion_matrix_full_wmu.csv')
    plot_confusion_matrix(best_flat_pred['EventType'].to_numpy(), best_flat_pred['PredictedEventType'].to_numpy(), labels, f"Flat full-WMU confusion matrix ({best_flat['EventModel']})", figures_dir / 'confusion_matrix_full_wmu.png')
    best_flat_pred.loc[~best_flat_pred['IsCorrect']].to_csv(reports_dir / 'misclassified_cases_full_wmu.csv', index=False)

    best_hier = hierarchical.metrics.iloc[0]
    best_hier_pred = _best_prediction_subset(hierarchical.predictions, best_hier)
    plot_confusion_matrix(best_hier_pred['EventType'].to_numpy(), best_hier_pred['PredictedEventType'].to_numpy(), labels, f"Hierarchical full-WMU confusion matrix ({best_hier['TriggerMethod']} + {best_hier['EventModel']})", figures_dir / 'confusion_matrix_hierarchical_full_wmu.png')
    write_confusion_matrix_csv(best_hier_pred['EventType'].to_numpy(), best_hier_pred['PredictedEventType'].to_numpy(), labels, reports_dir / 'confusion_matrix_hierarchical_full_wmu.csv')
    best_hier_pred.loc[~best_hier_pred['IsCorrect']].to_csv(reports_dir / 'misclassified_cases_hierarchical_full_wmu.csv', index=False)

    return EvaluationArtifacts(
        metrics=combined,
        predictions=pd.concat([flat.predictions, hierarchical.predictions], ignore_index=True),
        stratified_metrics=flat.stratified_metrics,
        feature_columns={'All_features': flat.feature_columns['All_features'] if flat.feature_columns else []},
        trigger_details=hierarchical.trigger_details,
    )


def evaluate_feature_ablation(by_case_wide: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> EvaluationArtifacts:
    metrics_frames = []
    pred_frames = []
    feature_columns_by_group: dict[str, list[str]] = {}
    trigger_frames = []
    groups = ['DV_energy_only', 'Voltage_time_only', 'Voltage_time_freq', 'Voltage_current', 'Voltage_current_unbalance_sequence', 'Impedance_added', 'All_features']
    for group_name in groups:
        flat = evaluate_models_for_group(by_case_wide, group_name, selected_models=['LogisticRegression', 'RandomForest'])
        hier = evaluate_hierarchical_for_group(by_case_wide, group_name, event_models=['RandomForest'], binary_trigger_models=['RandomForest'])
        metrics_frames.append(flat.metrics)
        metrics_frames.append(hier.metrics)
        pred_frames.append(flat.predictions)
        pred_frames.append(hier.predictions)
        feature_columns_by_group[group_name] = flat.feature_columns[group_name] if flat.feature_columns else []
        if hier.trigger_details is not None:
            trigger_frames.append(hier.trigger_details)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    trigger_details = pd.concat(trigger_frames, ignore_index=True) if trigger_frames else pd.DataFrame()
    metrics.to_csv(reports_dir / 'feature_ablation_metrics.csv', index=False)
    if not trigger_details.empty:
        trigger_details.to_csv(reports_dir / 'feature_ablation_trigger_debug.csv', index=False)

    used_columns_lines = []
    for group_name, columns in feature_columns_by_group.items():
        used_columns_lines.append(f'[{group_name}]')
        used_columns_lines.append(f'count={len(columns)}')
        used_columns_lines.extend(columns)
        used_columns_lines.append('')
    fingerprints = {group_name: hashlib.md5('\n'.join(feature_columns_by_group[group_name]).encode('utf-8')).hexdigest() for group_name in feature_columns_by_group}
    used_columns_lines.append('[matrix_fingerprints]')
    for group_name, fingerprint in fingerprints.items():
        used_columns_lines.append(f'{group_name}={fingerprint}')
    (reports_dir / 'feature_ablation_used_columns.txt').write_text('\n'.join(used_columns_lines) + '\n', encoding='utf-8')

    best_by_group_strategy = (
        metrics.sort_values(['FeatureGroup', 'Strategy', 'macro_f1', 'Normal_recall', 'LoadSwitch_recall'], ascending=[True, True, False, False, False])
        .groupby(['FeatureGroup', 'Strategy'], as_index=False)
        .first()
    )
    strategy_palette = {'flat': '#4c78a8', 'hierarchical': '#f58518'}
    for metric_name, filename, ylabel in [
        ('macro_f1', 'feature_ablation_macro_f1.png', 'Macro-F1'),
        ('Normal_recall', 'feature_ablation_normal_recall.png', 'Normal recall'),
        ('LoadSwitch_recall', 'feature_ablation_loadswitch_recall.png', 'LoadSwitch recall'),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        feature_groups = groups
        x = np.arange(len(feature_groups))
        width = 0.35
        for offset, strategy in [(-width / 2, 'flat'), (width / 2, 'hierarchical')]:
            subset = best_by_group_strategy.loc[best_by_group_strategy['Strategy'] == strategy].set_index('FeatureGroup').reindex(feature_groups)
            ax.bar(x + offset, subset[metric_name], width=width, label=strategy, color=strategy_palette[strategy])
        ax.set_xticks(x, feature_groups, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Feature ablation comparison ({ylabel})')
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=200)
        plt.close(fig)
    return EvaluationArtifacts(metrics=metrics, predictions=predictions, stratified_metrics=pd.DataFrame(), feature_columns=feature_columns_by_group, trigger_details=trigger_details)
