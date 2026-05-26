from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from wmu_project.waveform_features import select_feature_columns_by_group
from wmu_project.waveform_io import load_waveform_case
from wmu_project.waveform_utils import EVENT_CLASS_ORDER, detect_observed_buses, list_waveform_case_files, signal_columns_for_bus, to_local_path

sns.set_theme(style='whitegrid', context='talk')

CLASS_ORDER = [label for label in EVENT_CLASS_ORDER if label in {'Normal', 'LoadSwitch', 'SLG_Fault', 'ThreePhase_Fault'}]
CLASS_PALETTE = {
    'Normal': '#4c78a8',
    'LoadSwitch': '#f58518',
    'SLG_Fault': '#e45756',
    'ThreePhase_Fault': '#72b7b2',
}

CORE_CASE_FEATURES = [
    'max_dV_energy',
    'max_dI_energy',
    'max_sag',
    'max_HF_ratio',
    'max_Res_ratio',
    'max_V0_ratio',
    'max_I0_ratio',
    'max_V_unbalance',
    'max_I_unbalance',
    'max_Z_drop_ratio',
    'min_liss_corr_abs_min',
    'max_liss_area_norm_min',
]

CORE_FEATURE_LABELS = {
    'max_dV_energy': 'max_dV_energy',
    'max_dI_energy': 'max_dI_energy',
    'max_sag': 'max_sag',
    'max_HF_ratio': 'max_HF_ratio',
    'max_Res_ratio': 'max_Res_ratio',
    'max_V0_ratio': 'max_V0_ratio',
    'max_I0_ratio': 'max_I0_ratio',
    'max_V_unbalance': 'max_V_unbalance',
    'max_I_unbalance': 'max_I_unbalance',
    'max_Z_drop_ratio': 'max_Z_drop_ratio',
    'min_liss_corr_abs_min': 'min liss_corr_abs_min',
    'max_liss_area_norm_min': 'max liss_area_norm_min',
}

SCATTER_SPECS = [
    ('max_dV_energy', 'max_dI_energy', 'scatter_dv_di.png', 'max_dV_energy vs max_dI_energy'),
    ('max_sag', 'max_dI_energy', 'scatter_sag_di.png', 'max_sag vs max_dI_energy'),
    ('max_I0_ratio', 'max_V0_ratio', 'scatter_i0_v0.png', 'max_I0_ratio vs max_V0_ratio'),
    ('max_Res_ratio', 'max_HF_ratio', 'scatter_res_hf.png', 'max_Res_ratio vs max_HF_ratio'),
    ('max_Z_drop_ratio', 'max_dI_energy', 'scatter_zdrop_di.png', 'max_Z_drop_ratio vs max_dI_energy'),
    ('max_I_unbalance', 'max_V_unbalance', 'scatter_unbalance.png', 'max_I_unbalance vs max_V_unbalance'),
]

MISCLASS_FEATURES = [
    'max_dV_energy',
    'max_dI_energy',
    'max_sag',
    'max_V0_ratio',
    'max_I0_ratio',
    'max_Z_drop_ratio',
    'max_V_unbalance',
    'max_I_unbalance',
]

CORRELATION_FEATURES = [
    'max_dV_energy',
    'max_dI_energy',
    'max_sag',
    'max_HF_ratio',
    'max_Res_ratio',
    'max_V0_ratio',
    'max_I0_ratio',
    'max_V_unbalance',
    'max_I_unbalance',
    'max_Z_drop_ratio',
    'min_liss_corr_abs_min',
    'max_liss_area_norm_min',
]

HEATMAP_SPECS = [
    ('SLG_Fault', 'dV_energy_3ph_max', 'heatmap_slg_dv_energy.png', 'SLG_Fault mean dV_energy_3ph_max by TargetBus × ObservedBus'),
    ('ThreePhase_Fault', 'dV_energy_3ph_max', 'heatmap_threephase_dv_energy.png', 'ThreePhase_Fault mean dV_energy_3ph_max by TargetBus × ObservedBus'),
    ('LoadSwitch', 'dV_energy_3ph_max', 'heatmap_loadswitch_dv_energy.png', 'LoadSwitch mean dV_energy_3ph_max by TargetBus × ObservedBus'),
    ('SLG_Fault', 'I0_ratio', 'heatmap_slg_i0_ratio.png', 'SLG_Fault mean I0_ratio by TargetBus × ObservedBus'),
    ('ThreePhase_Fault', 'dI_energy_3ph_max', 'heatmap_threephase_di_energy.png', 'ThreePhase_Fault mean dI_energy_3ph_max by TargetBus × ObservedBus'),
    ('LoadSwitch', 'dI_energy_3ph_max', 'heatmap_loadswitch_di_energy.png', 'LoadSwitch mean dI_energy_3ph_max by TargetBus × ObservedBus'),
]

REPRESENTATIVE_CASES = {
    'waveform_representative_normal.png': ['Normal_Case01'],
    'waveform_representative_loadswitch.png': ['LoadSwitch_Bus02', 'LoadSwitch_Bus10'],
    'waveform_representative_slg.png': ['SLG_Fault_Bus02', 'SLG_Fault_Bus10'],
    'waveform_representative_threephase.png': ['ThreePhase_Fault_Bus02', 'ThreePhase_Fault_Bus10'],
}

REPRESENTATIVE_WINDOWS = {
    'Normal': (0.45, 0.65),
    'LoadSwitch': (0.15, 0.35),
    'SLG_Fault': (0.45, 0.65),
    'ThreePhase_Fault': (0.45, 0.65),
}

BUS_FEATURE_RE = re.compile(r'^Bus(?P<bus>\d+)__')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate waveform feature diagnostics figures and reports')
    parser.add_argument('--data-dir', required=True, help='Directory containing feature_table_by_*.csv')
    parser.add_argument('--raw-dir', required=True, help='Directory containing raw waveform csv/xlsx files')
    parser.add_argument('--results-dir', required=True, help='Output directory for reports and figures')
    return parser.parse_args()


def ensure_dirs(results_dir: Path) -> tuple[Path, Path]:
    reports_dir = results_dir / 'reports'
    figures_dir = results_dir / 'figures'
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir, figures_dir


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_tables(data_dir: Path, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_bus = pd.read_csv(data_dir / 'feature_table_by_bus.csv')
    by_case = pd.read_csv(data_dir / 'feature_table_by_case.csv')
    by_case_wide = pd.read_csv(data_dir / 'feature_table_by_case_wide.csv')
    mis_path = first_existing([
        results_dir / 'reports' / 'misclassified_cases_full_wmu.csv',
        data_dir / 'reports' / 'misclassified_cases_full_wmu.csv',
        ROOT / 'results' / 'waveform_event_analysis' / 'reports' / 'misclassified_cases_full_wmu.csv',
    ])
    if mis_path is None:
        raise FileNotFoundError('misclassified_cases_full_wmu.csv not found in results-dir/reports or data-dir/reports')
    misclassified = pd.read_csv(mis_path)
    for frame in (by_bus, by_case, by_case_wide, misclassified):
        if 'EventType' in frame.columns:
            frame['EventType'] = pd.Categorical(frame['EventType'].astype(str), CLASS_ORDER, ordered=True)
    return by_bus, by_case, by_case_wide, misclassified


def build_case_enriched(by_case: pd.DataFrame, by_bus: pd.DataFrame) -> pd.DataFrame:
    agg = (
        by_bus.groupby('CaseName', dropna=False)
        .agg(
            min_liss_corr_abs_min=('liss_corr_abs_min', 'min'),
            max_liss_area_norm_min=('liss_area_norm_min', 'max'),
        )
        .reset_index()
    )
    enriched = by_case.merge(agg, on='CaseName', how='left')
    enriched['EventType'] = pd.Categorical(enriched['EventType'].astype(str), CLASS_ORDER, ordered=True)
    return enriched.sort_values(['EventType', 'TargetBus', 'CaseName']).reset_index(drop=True)


def available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in df.columns]


def save_feature_distribution_summary(case_df: pd.DataFrame, features: list[str], out_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        for event_type in CLASS_ORDER:
            subset = pd.to_numeric(case_df.loc[case_df['EventType'].astype(str) == event_type, feature], errors='coerce').dropna()
            if subset.empty:
                continue
            rows.append({
                'Feature': feature,
                'EventType': event_type,
                'count': int(subset.size),
                'mean': float(subset.mean()),
                'std': float(subset.std(ddof=1)) if subset.size > 1 else 0.0,
                'median': float(subset.median()),
                'q25': float(subset.quantile(0.25)),
                'q75': float(subset.quantile(0.75)),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)
    return summary


def _facet_plot(case_df: pd.DataFrame, features: list[str], kind: str, out_path: Path, title: str) -> None:
    n = len(features)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5), squeeze=False)
    axes_flat = axes.flatten()
    for idx, feature in enumerate(features):
        ax = axes_flat[idx]
        plot_df = case_df[['EventType', feature]].copy().rename(columns={feature: 'Value'})
        plot_df = plot_df.dropna()
        if kind == 'box':
            sns.boxplot(data=plot_df, x='EventType', y='Value', hue='EventType', order=CLASS_ORDER, hue_order=CLASS_ORDER, palette=CLASS_PALETTE, ax=ax, fliersize=2, dodge=False, legend=False)
            sns.stripplot(data=plot_df, x='EventType', y='Value', order=CLASS_ORDER, ax=ax, color='black', size=3, alpha=0.45)
        else:
            sns.violinplot(data=plot_df, x='EventType', y='Value', hue='EventType', order=CLASS_ORDER, hue_order=CLASS_ORDER, palette=CLASS_PALETTE, ax=ax, inner='quartile', cut=0, legend=False)
            sns.stripplot(data=plot_df, x='EventType', y='Value', order=CLASS_ORDER, ax=ax, color='black', size=2.5, alpha=0.25)
        ax.set_title(CORE_FEATURE_LABELS.get(feature, feature))
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=25)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    for ax in axes_flat[n:]:
        ax.axis('off')
    fig.suptitle(title, y=1.01, fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_feature_distributions(case_df: pd.DataFrame, features: list[str], figures_dir: Path) -> None:
    _facet_plot(case_df, features, 'box', figures_dir / 'feature_boxplot_core_features.png', 'Core feature distributions by EventType (boxplot)')
    _facet_plot(case_df, features, 'violin', figures_dir / 'feature_violin_core_features.png', 'Core feature distributions by EventType (violin)')


def plot_scatter(case_df: pd.DataFrame, figures_dir: Path) -> None:
    for x_feature, y_feature, filename, title in SCATTER_SPECS:
        if x_feature not in case_df.columns or y_feature not in case_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=case_df,
            x=x_feature,
            y=y_feature,
            hue='EventType',
            hue_order=CLASS_ORDER,
            palette=CLASS_PALETTE,
            s=90,
            ax=ax,
        )
        ax.set_title(title)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))
        ax.legend(title='EventType', bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=220, bbox_inches='tight')
        plt.close(fig)


def percentile_rank(values: pd.Series, value: float) -> float:
    series = pd.to_numeric(values, errors='coerce').dropna()
    if series.empty:
        return float('nan')
    return float(np.mean(series <= value))


def normal_misclassification_analysis(case_df: pd.DataFrame, misclassified: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    mis = misclassified.copy()
    mis['EventType'] = mis['EventType'].astype(str)
    mis['PredictedEventType'] = mis['PredictedEventType'].astype(str)
    target_cases = mis.loc[(mis['EventType'] == 'Normal') & (mis['PredictedEventType'] == 'SLG_Fault'), 'CaseName'].astype(str).tolist()
    selected_features = available_features(case_df, MISCLASS_FEATURES)
    mis_rows: list[dict[str, object]] = []
    mis_case_df = case_df.loc[case_df['CaseName'].astype(str).isin(target_cases)].copy()
    for _, row in mis_case_df.iterrows():
        for feature in selected_features:
            value = float(row[feature]) if pd.notna(row[feature]) else float('nan')
            record: dict[str, object] = {
                'CaseName': row['CaseName'],
                'EventType': row['EventType'],
                'Feature': feature,
                'Value': value,
            }
            for event_type in CLASS_ORDER:
                ref = case_df.loc[case_df['EventType'].astype(str) == event_type, feature]
                record[f'percentile_in_{event_type}'] = percentile_rank(ref, value)
                record[f'median_{event_type}'] = float(pd.to_numeric(ref, errors='coerce').median()) if not ref.dropna().empty else float('nan')
            mis_rows.append(record)
    mis_values = pd.DataFrame(mis_rows)
    mis_values.to_csv(reports_dir / 'normal_misclassification_feature_values.csv', index=False)

    if selected_features:
        cols = 2
        rows = math.ceil(len(selected_features) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5), squeeze=False)
        axes_flat = axes.flatten()
        highlight_cases = case_df.loc[case_df['CaseName'].astype(str).isin(target_cases), ['CaseName'] + selected_features].copy()
        for idx, feature in enumerate(selected_features):
            ax = axes_flat[idx]
            plot_df = case_df[['EventType', feature]].dropna().copy()
            sns.boxplot(data=plot_df, x='EventType', y=feature, hue='EventType', order=CLASS_ORDER, hue_order=CLASS_ORDER, palette=CLASS_PALETTE, ax=ax, fliersize=2, dodge=False, legend=False)
            sns.stripplot(data=plot_df, x='EventType', y=feature, order=CLASS_ORDER, ax=ax, color='black', size=2.5, alpha=0.25)
            for _, mis_row in highlight_cases.iterrows():
                y_val = mis_row[feature]
                if pd.isna(y_val):
                    continue
                x_pos = CLASS_ORDER.index('Normal')
                ax.scatter(x_pos, y_val, marker='*', s=220, color='gold', edgecolor='black', linewidth=0.8, zorder=5)
                ax.text(x_pos + 0.05, y_val, str(mis_row['CaseName']).replace('Normal_', ''), fontsize=8, va='center')
            ax.set_title(CORE_FEATURE_LABELS.get(feature, feature))
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=25)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        for ax in axes_flat[len(selected_features):]:
            ax.axis('off')
        legend_handles = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=14, label='Normal cases misclassified as SLG_Fault')
        ]
        fig.legend(handles=legend_handles, loc='upper right')
        fig.suptitle('Normal → SLG_Fault misclassification feature positions', y=1.01, fontsize=18)
        fig.tight_layout()
        fig.savefig(figures_dir / 'normal_misclassification_core_features.png', dpi=220, bbox_inches='tight')
        plt.close(fig)
    return mis_values


def correlation_heatmap(case_df: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    features = available_features(case_df, CORRELATION_FEATURES)
    corr = case_df[features].corr(numeric_only=True)
    corr.to_csv(reports_dir / 'feature_correlation_matrix.csv')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0.0, square=True, ax=ax)
    ax.set_title('Case-level core feature correlation heatmap')
    fig.tight_layout()
    fig.savefig(figures_dir / 'feature_correlation_heatmap.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    return corr


def randomforest_importance(by_case_wide: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = select_feature_columns_by_group(by_case_wide, 'All_features')
    X = by_case_wide[feature_columns]
    y = by_case_wide['EventType'].astype(str)
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1,
    )
    model.fit(X_imp, y)
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    importance.to_csv(reports_dir / 'randomforest_feature_importance.csv', index=False)

    top30 = importance.head(30).sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(top30['feature'], top30['importance'], color='#4c78a8')
    ax.set_title('RandomForest top 30 feature importance')
    ax.set_xlabel('Importance')
    fig.tight_layout()
    fig.savefig(figures_dir / 'randomforest_top30_feature_importance.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    bus_rows: list[dict[str, object]] = []
    for _, row in importance.head(30).iterrows():
        match = BUS_FEATURE_RE.match(str(row['feature']))
        if match:
            bus_rows.append({'ObservedBus': int(match.group('bus')), 'feature': row['feature'], 'importance': row['importance']})
        else:
            bus_rows.append({'ObservedBus': 'CaseSummary', 'feature': row['feature'], 'importance': row['importance']})
    bus_df = pd.DataFrame(bus_rows)
    bus_count = (
        bus_df.groupby('ObservedBus', dropna=False)
        .agg(feature_count=('feature', 'count'), total_importance=('importance', 'sum'))
        .reset_index()
        .sort_values(['feature_count', 'total_importance'], ascending=[False, False])
    )
    bus_count.to_csv(reports_dir / 'important_wmu_bus_count.csv', index=False)

    plot_df = bus_count.loc[bus_count['ObservedBus'] != 'CaseSummary'].copy()
    if not plot_df.empty:
        plot_df['ObservedBusLabel'] = plot_df['ObservedBus'].map(lambda x: f'Bus{int(x):02d}')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=plot_df, x='ObservedBusLabel', y='feature_count', color='#f58518', ax=ax)
        ax.set_title('Observed bus frequency among RandomForest top 30 features')
        ax.set_xlabel('ObservedBus')
        ax.set_ylabel('Top-30 feature count')
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        fig.savefig(figures_dir / 'important_wmu_bus_count.png', dpi=220, bbox_inches='tight')
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No bus-specific features in top 30', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(figures_dir / 'important_wmu_bus_count.png', dpi=220, bbox_inches='tight')
        plt.close(fig)
    return importance, bus_count


def spatial_heatmaps(by_bus: pd.DataFrame, figures_dir: Path) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for event_type, feature, filename, title in HEATMAP_SPECS:
        subset = by_bus.loc[by_bus['EventType'].astype(str) == event_type, ['TargetBus', 'ObservedBus', feature]].copy()
        subset = subset.dropna(subset=['TargetBus', 'ObservedBus', feature])
        subset['TargetBus'] = subset['TargetBus'].astype(int)
        subset['ObservedBus'] = subset['ObservedBus'].astype(int)
        pivot = subset.pivot_table(index='TargetBus', columns='ObservedBus', values=feature, aggfunc='mean').sort_index().reindex(sorted(subset['ObservedBus'].unique()), axis=1)
        outputs[filename] = pivot
        fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(pivot.index))))
        sns.heatmap(pivot, cmap='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('ObservedBus')
        ax.set_ylabel('TargetBus')
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=220, bbox_inches='tight')
        plt.close(fig)
    return outputs


def pick_case_metadata(raw_dir: Path, preferred_names: list[str]):
    cases = {meta.case_name: meta for meta in list_waveform_case_files(raw_dir)}
    for name in preferred_names:
        if name in cases:
            return cases[name]
    return None


def representative_waveform_plots(raw_dir: Path, case_df: pd.DataFrame, figures_dir: Path) -> list[str]:
    generated: list[str] = []
    for filename, preferred_names in REPRESENTATIVE_CASES.items():
        meta = pick_case_metadata(raw_dir, preferred_names)
        if meta is None:
            continue
        df = load_waveform_case(meta)
        case_row = case_df.loc[case_df['CaseName'].astype(str) == meta.case_name].iloc[0]
        observed_buses = detect_observed_buses(df.columns)
        bus = None
        if meta.target_bus_int is not None and meta.target_bus_int in observed_buses:
            bus = meta.target_bus_int
        else:
            star_bus = int(case_row['star_bus_dV']) if pd.notna(case_row['star_bus_dV']) else observed_buses[0]
            bus = star_bus if star_bus in observed_buses else observed_buses[0]
        t0, t1 = REPRESENTATIVE_WINDOWS[meta.event_type]
        window_df = df.loc[(df['Time'] >= t0) & (df['Time'] <= t1)].copy()
        cols = signal_columns_for_bus(bus)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for phase, color in zip(['A', 'B', 'C'], ['#4c78a8', '#f58518', '#54a24b']):
            axes[0].plot(window_df['Time'], window_df[cols[f'V{phase}']], label=f'V{phase}', linewidth=1.4, color=color)
            axes[1].plot(window_df['Time'], window_df[cols[f'I{phase}']], label=f'I{phase}', linewidth=1.4, color=color)
        for ax in axes:
            ax.axvline(meta.event_time, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
            ax.grid(True, alpha=0.25)
        axes[0].set_title(f'{meta.case_name} at observed bus {bus:02d} (EventType={meta.event_type})')
        axes[0].set_ylabel('Voltage')
        axes[1].set_ylabel('Current')
        axes[1].set_xlabel('Time [s]')
        axes[0].legend(ncol=3, loc='upper right')
        axes[1].legend(ncol=3, loc='upper right')
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=220, bbox_inches='tight')
        plt.close(fig)
        generated.append(filename)
    return generated


def compute_separation_summary(case_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = case_df[features]
    y = case_df['EventType'].astype(str)
    X_imp = SimpleImputer(strategy='median').fit_transform(X)
    f_stat, p_val = f_classif(X_imp, y)
    overall = pd.DataFrame({'Feature': features, 'anova_f': f_stat, 'anova_p': p_val}).sort_values('anova_f', ascending=False).reset_index(drop=True)

    fault_mask = y.isin(['SLG_Fault', 'ThreePhase_Fault'])
    compare_rows: list[dict[str, object]] = []
    for feature in features:
        ls = pd.to_numeric(case_df.loc[y == 'LoadSwitch', feature], errors='coerce').dropna()
        fault = pd.to_numeric(case_df.loc[fault_mask, feature], errors='coerce').dropna()
        if ls.empty or fault.empty:
            continue
        pooled = np.sqrt((ls.var(ddof=1) if len(ls) > 1 else 0.0 + fault.var(ddof=1) if len(fault) > 1 else 0.0) / 2.0)
        effect = abs(ls.mean() - fault.mean()) / pooled if pooled and np.isfinite(pooled) else 0.0
        compare_rows.append({
            'Feature': feature,
            'loadswitch_mean': float(ls.mean()),
            'fault_mean': float(fault.mean()),
            'abs_effect_size': float(effect),
        })
    comparison = pd.DataFrame(compare_rows).sort_values('abs_effect_size', ascending=False).reset_index(drop=True)
    return overall, comparison


def top_correlations(corr: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1:]:
            rows.append({'feature_a': col_a, 'feature_b': col_b, 'corr_abs': abs(float(corr.loc[col_a, col_b])), 'corr': float(corr.loc[col_a, col_b])})
    return pd.DataFrame(rows).sort_values('corr_abs', ascending=False).head(n).reset_index(drop=True)


def summarize_normal_misclassification(mis_values: pd.DataFrame) -> str:
    if mis_values.empty:
        return '- No Normal → SLG_Fault cases were found in the supplied misclassification report.'
    first_rows = mis_values.groupby('Feature', as_index=False).first()
    overlap_features = []
    slg_like_features = []
    for _, row in first_rows.iterrows():
        value = float(row['Value'])
        slg_median = float(row['median_SLG_Fault'])
        normal_median = float(row['median_Normal'])
        ls_median = float(row['median_LoadSwitch'])
        tp_median = float(row['median_ThreePhase_Fault'])
        tol = max(1e-9, 1e-6 * max(1.0, abs(value), abs(slg_median), abs(normal_median)))
        if abs(value - slg_median) <= tol and abs(normal_median - slg_median) <= tol:
            overlap_features.append(str(row['Feature']))
        slg_gap = abs(value - slg_median)
        ls_gap = abs(value - ls_median)
        tp_gap = abs(value - tp_median)
        if slg_gap <= min(ls_gap, tp_gap):
            slg_like_features.append(str(row['Feature']))
    lines = ['- Only 3 Normal cases exist, so Normal false-alarm interpretation is unstable.']
    if overlap_features:
        lines.append('- Normal and SLG_Fault are numerically overlapped on several core case-level features: ' + ', '.join(overlap_features[:6]) + '.')
    if slg_like_features:
        lines.append('- The misclassified Normal cases lie closer to the SLG_Fault medians than to LoadSwitch or ThreePhase medians for: ' + ', '.join(slg_like_features[:6]) + '.')
    for feature in overlap_features[:3]:
        row = first_rows.loc[first_rows['Feature'] == feature].iloc[0]
        lines.append(f"- {feature}: Normal value={row['Value']:.4g}, SLG median={row['median_SLG_Fault']:.4g}, LoadSwitch median={row['median_LoadSwitch']:.4g}, ThreePhase median={row['median_ThreePhase_Fault']:.4g}.")
    return '\n'.join(lines)


def write_summary_md(
    out_path: Path,
    case_df: pd.DataFrame,
    dist_summary: pd.DataFrame,
    separation_overall: pd.DataFrame,
    separation_loadswitch_fault: pd.DataFrame,
    mis_values: pd.DataFrame,
    corr: pd.DataFrame,
    rf_importance: pd.DataFrame,
    bus_count: pd.DataFrame,
    heatmaps: dict[str, pd.DataFrame],
) -> None:
    class_counts = case_df['EventType'].astype(str).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    top_sep = separation_overall.head(6)['Feature'].tolist()
    top_ls_fault = separation_loadswitch_fault.head(6)['Feature'].tolist() if not separation_loadswitch_fault.empty else []
    top_corr = top_correlations(corr)
    top_rf = rf_importance.head(10)
    top_bus = bus_count.head(6)

    spatial_lines = []
    for filename, pivot in heatmaps.items():
        if pivot.empty:
            continue
        max_loc = np.unravel_index(np.nanargmax(pivot.to_numpy()), pivot.shape)
        target_bus = pivot.index[max_loc[0]]
        observed_bus = pivot.columns[max_loc[1]]
        spatial_lines.append(f'- {filename}: strongest average response at TargetBus {int(target_bus)} / ObservedBus {int(observed_bus)}.')

    feature_mean_lines = []
    for feature in top_sep[:4]:
        view = dist_summary.loc[dist_summary['Feature'] == feature, ['EventType', 'mean']].copy()
        if view.empty:
            continue
        best = view.sort_values('mean', ascending=False).iloc[0]
        feature_mean_lines.append(f"- {feature}: highest mean response in {best['EventType']} ({best['mean']:.4g}).")

    corr_lines = [f"- {row.feature_a} vs {row.feature_b}: corr={row.corr:.3f}" for row in top_corr.itertuples(index=False)]
    rf_lines = [f"- {row.feature}: {row.importance:.4f}" for row in top_rf.itertuples(index=False)]
    bus_lines = [f"- {row.ObservedBus}: top-feature count={row.feature_count}, summed importance={row.total_importance:.4f}" for row in top_bus.itertuples(index=False)]

    text = f"""# Feature Diagnostics Summary

## 1. 분석 목적
현재 83개 WMU waveform dataset에서 추출한 feature들이 **Normal / LoadSwitch / SLG_Fault / ThreePhase_Fault**를 어떻게 구분하는지 시각적으로 해석하는 데 초점을 두었다. 본 결과는 최종 classification 성능 주장보다 **feature separability, dataset limitation, 오분류 원인 분석**을 우선한다.

## 2. 사용 데이터셋 83개 구성
- Total: {int(class_counts.sum())}
- Normal: {class_counts['Normal']}
- LoadSwitch: {class_counts['LoadSwitch']}
- SLG_Fault: {class_counts['SLG_Fault']}
- ThreePhase_Fault: {class_counts['ThreePhase_Fault']}
- Quality summary from prior run: OK 83 / WARNING 0 / FAILED 0

## 3. Feature 분포 분석 요약
주요 case-level/core feature에 대해 EventType별 boxplot/violin plot과 요약 통계를 생성했다.

가장 큰 class-level separation을 보인 핵심 feature(ANOVA F-score 기준):
{chr(10).join(f'- {feature}' for feature in top_sep)}

분포 평균 기준 관찰:
{chr(10).join(feature_mean_lines) if feature_mean_lines else '- Summary unavailable'}

## 4. LoadSwitch와 Fault 구분에 유효해 보이는 feature
LoadSwitch vs Fault(SLG + ThreePhase) 평균 차이를 표준화한 effect size 기준 상위 feature:
{chr(10).join(f'- {feature}' for feature in top_ls_fault) if top_ls_fault else '- No comparison available'}

해석적으로는 dV/dI disturbance energy, sag, sequence/unbalance, impedance-drop 계열이 LoadSwitch와 Fault를 가장 직접적으로 가르는 축으로 보인다.

## 5. Normal 오분류 원인 분석
{summarize_normal_misclassification(mis_values)}
- 현재 데이터에서는 Normal이 3개뿐이므로, Normal false alarm의 일반화 해석은 매우 불안정하다.
- 그럼에도 misclassified Normal의 특정 feature 값이 SLG_Fault 분포 내부 또는 경계에 들어가는지 feature 관점에서 직접 확인할 수 있다.

## 6. Correlation / importance 결과 요약
상관이 특히 큰 feature 쌍:
{chr(10).join(corr_lines) if corr_lines else '- Correlation summary unavailable'}

RandomForest 상위 중요 feature:
{chr(10).join(rf_lines) if rf_lines else '- RF importance unavailable'}

Top-30 중요 feature의 ObservedBus 분포:
{chr(10).join(bus_lines) if bus_lines else '- Bus aggregation unavailable'}

## 7. Spatial heatmap 해석
TargetBus × ObservedBus mean-response heatmap으로 disturbance spatial footprint를 요약했다.
{chr(10).join(spatial_lines) if spatial_lines else '- Heatmap summary unavailable'}
- LoadSwitch heatmap의 TargetBus index가 20개만 나오는 것은 load bus만 생성된 현재 dataset 구성상 정상이다.

## 8. 현재 데이터에서 DV_energy가 지배적인 이유에 대한 해석
- 현재 dataset은 event timing이 고정되어 있고 disturbance type별 파형 변화 크기가 크기 때문에, one-cycle difference 기반 dV_energy가 가장 직접적인 trigger strength를 제공한다.
- Fault 계열은 전압 파형 붕괴/급변이 크고, LoadSwitch는 상대적으로 전류 변화 비중이 커서 dV_energy 및 dI_energy가 먼저 class boundary를 만든다.
- 데이터 수가 작고 조건 다양성이 제한되어 있으므로, 강한 energy feature 하나가 다른 미세 feature보다 훨씬 안정적으로 작동한다.

## 9. 추가 feature가 아직 성능 향상으로 이어지지 않은 이유에 대한 가설
- 현재 83개 데이터는 이미 separability가 높아 richer feature가 macro-F1을 더 올릴 여지가 작다.
- Normal class가 3개뿐이라 sequence/lissajous/impedance feature의 장점이 안정적으로 검증되지 않는다.
- Fault resistance, inception angle, clearing time, switching intensity variation이 부족해 고급 feature가 드러날 조건 변화가 충분하지 않다.
- 일부 feature는 dV/dI energy와 강하게 상관되어 정보 중복이 발생한다.

## 10. 다음 실험 제안
- LoadSwitch 강도를 5%, 15%, 25% 이상으로 늘려 switching manifold를 확장한다.
- Fault resistance / clearing time / inception angle variation을 추가해 fault morphology 다양성을 늘린다.
- Normal baseline을 충분히 늘려 false alarm 해석의 신뢰구간을 확보한다.
- 반복 시뮬레이션을 추가해 case-level train/test split이 가능한 dataset으로 확장한다.
- WMU bus importance를 target-bus distance 또는 network topology와 함께 해석해 spatial diagnostics를 강화한다.
"""
    out_path.write_text(text, encoding='utf-8')


def main() -> None:
    args = parse_args()
    data_dir = to_local_path(args.data_dir)
    raw_dir = to_local_path(args.raw_dir)
    results_dir = to_local_path(args.results_dir)
    reports_dir, figures_dir = ensure_dirs(results_dir)

    by_bus, by_case, by_case_wide, misclassified = load_tables(data_dir, results_dir)
    case_df = build_case_enriched(by_case, by_bus)
    core_features = available_features(case_df, CORE_CASE_FEATURES)

    dist_summary = save_feature_distribution_summary(case_df, core_features, reports_dir / 'feature_distribution_summary.csv')
    plot_feature_distributions(case_df, core_features, figures_dir)
    plot_scatter(case_df, figures_dir)
    mis_values = normal_misclassification_analysis(case_df, misclassified, reports_dir, figures_dir)
    corr = correlation_heatmap(case_df, reports_dir, figures_dir)
    rf_importance, bus_count = randomforest_importance(by_case_wide, reports_dir, figures_dir)
    heatmaps = spatial_heatmaps(by_bus, figures_dir)
    generated_waveforms = representative_waveform_plots(raw_dir, case_df, figures_dir)
    separation_overall, separation_loadswitch_fault = compute_separation_summary(case_df, core_features)
    write_summary_md(
        reports_dir / 'feature_diagnostics_summary.md',
        case_df,
        dist_summary,
        separation_overall,
        separation_loadswitch_fault,
        mis_values,
        corr,
        rf_importance,
        bus_count,
        heatmaps,
    )

    print('Generated reports:')
    for path in sorted(reports_dir.glob('*')):
        print(path)
    print('Generated figures:')
    for path in sorted(figures_dir.glob('*')):
        print(path)
    print(f'Representative waveform figures generated: {generated_waveforms}')


if __name__ == '__main__':
    main()
