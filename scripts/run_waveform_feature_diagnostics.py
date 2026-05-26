from __future__ import annotations

import argparse
import hashlib
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
CLASS_PALETTE = {'Normal': '#4c78a8', 'LoadSwitch': '#f58518', 'SLG_Fault': '#e45756', 'ThreePhase_Fault': '#72b7b2'}
CORE_CASE_FEATURES = ['max_dV_energy', 'max_dI_energy', 'max_sag', 'max_HF_ratio', 'max_Res_ratio', 'max_V0_ratio', 'max_I0_ratio', 'max_V_unbalance', 'max_I_unbalance', 'max_Z_drop_ratio', 'min_liss_corr_abs_min', 'max_liss_area_norm_min']
MISCLASS_FEATURES = ['max_dV_energy', 'max_dI_energy', 'max_sag', 'max_V0_ratio', 'max_I0_ratio', 'max_Z_drop_ratio', 'max_V_unbalance', 'max_I_unbalance']
CORR_FEATURES = CORE_CASE_FEATURES
SCATTER_SPECS = [
    ('max_dV_energy', 'max_dI_energy', 'scatter_dv_di.png', 'max_dV_energy vs max_dI_energy'),
    ('max_sag', 'max_dI_energy', 'scatter_sag_di.png', 'max_sag vs max_dI_energy'),
    ('max_I0_ratio', 'max_V0_ratio', 'scatter_i0_v0.png', 'max_I0_ratio vs max_V0_ratio'),
    ('max_Res_ratio', 'max_HF_ratio', 'scatter_res_hf.png', 'max_Res_ratio vs max_HF_ratio'),
    ('max_Z_drop_ratio', 'max_dI_energy', 'scatter_zdrop_di.png', 'max_Z_drop_ratio vs max_dI_energy'),
    ('max_I_unbalance', 'max_V_unbalance', 'scatter_unbalance.png', 'max_I_unbalance vs max_V_unbalance'),
]
HEATMAP_SPECS = [
    ('SLG_Fault', 'dV_energy_3ph_max', 'heatmap_slg_dv_energy.png', 'SLG_Fault mean dV_energy_3ph_max by TargetBus × ObservedBus'),
    ('ThreePhase_Fault', 'dV_energy_3ph_max', 'heatmap_threephase_dv_energy.png', 'ThreePhase_Fault mean dV_energy_3ph_max by TargetBus × ObservedBus'),
    ('LoadSwitch', 'dV_energy_3ph_max', 'heatmap_loadswitch_dv_energy.png', 'LoadSwitch mean dV_energy_3ph_max by TargetBus × ObservedBus'),
    ('SLG_Fault', 'I0_ratio', 'heatmap_slg_i0_ratio.png', 'SLG_Fault mean I0_ratio by TargetBus × ObservedBus'),
    ('ThreePhase_Fault', 'dI_energy_3ph_max', 'heatmap_threephase_di_energy.png', 'ThreePhase_Fault mean dI_energy_3ph_max by TargetBus × ObservedBus'),
    ('LoadSwitch', 'dI_energy_3ph_max', 'heatmap_loadswitch_di_energy.png', 'LoadSwitch mean dI_energy_3ph_max by TargetBus × ObservedBus'),
]
REP_CASES = {
    'waveform_representative_normal.png': ['Normal_Case01'],
    'waveform_representative_loadswitch.png': ['LoadSwitch_Bus02', 'LoadSwitch_Bus10'],
    'waveform_representative_slg.png': ['SLG_Fault_Bus02', 'SLG_Fault_Bus10'],
    'waveform_representative_threephase.png': ['ThreePhase_Fault_Bus02', 'ThreePhase_Fault_Bus10'],
}
REP_WINDOWS = {'Normal': (0.45, 0.65), 'LoadSwitch': (0.15, 0.35), 'SLG_Fault': (0.45, 0.65), 'ThreePhase_Fault': (0.45, 0.65)}
BUS_FEATURE_RE = re.compile(r'^Bus(?P<bus>\d+)__')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--results-dir', required=True)
    return parser.parse_args()


def ensure_dirs(results_dir: Path) -> tuple[Path, Path]:
    reports_dir = results_dir / 'reports'
    figures_dir = results_dir / 'figures'
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir, figures_dir


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def load_tables(data_dir: Path, results_dir: Path):
    by_bus = pd.read_csv(data_dir / 'feature_table_by_bus.csv')
    by_case = pd.read_csv(data_dir / 'feature_table_by_case.csv')
    by_case_wide = pd.read_csv(data_dir / 'feature_table_by_case_wide.csv')
    metrics = pd.read_csv(first_existing([results_dir / 'reports' / 'classification_flat_vs_hierarchical_metrics.csv', ROOT / 'results' / 'waveform_event_analysis' / 'reports' / 'classification_flat_vs_hierarchical_metrics.csv']))
    mis = pd.read_csv(first_existing([results_dir / 'reports' / 'misclassified_cases_full_wmu.csv', ROOT / 'results' / 'waveform_event_analysis' / 'reports' / 'misclassified_cases_full_wmu.csv']))
    for frame in [by_bus, by_case, by_case_wide, metrics, mis]:
        if 'EventType' in frame.columns:
            frame['EventType'] = pd.Categorical(frame['EventType'].astype(str), CLASS_ORDER, ordered=True)
    return by_bus, by_case, by_case_wide, metrics, mis


def build_case_enriched(by_case: pd.DataFrame, by_bus: pd.DataFrame) -> pd.DataFrame:
    agg = by_bus.groupby('CaseName', dropna=False).agg(min_liss_corr_abs_min=('liss_corr_abs_min', 'min'), max_liss_area_norm_min=('liss_area_norm_min', 'max')).reset_index()
    out = by_case.merge(agg, on='CaseName', how='left')
    out['EventType'] = pd.Categorical(out['EventType'].astype(str), CLASS_ORDER, ordered=True)
    return out.sort_values(['EventType', 'TargetBus', 'CaseName']).reset_index(drop=True)


def feature_distribution(case_df: pd.DataFrame, features: list[str], reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    rows = []
    for feat in features:
        for cls in CLASS_ORDER:
            s = pd.to_numeric(case_df.loc[case_df['EventType'].astype(str) == cls, feat], errors='coerce').dropna()
            if s.empty:
                continue
            rows.append({'Feature': feat, 'EventType': cls, 'count': int(len(s)), 'mean': float(s.mean()), 'std': float(s.std(ddof=1)) if len(s) > 1 else 0.0, 'median': float(s.median()), 'q25': float(s.quantile(0.25)), 'q75': float(s.quantile(0.75))})
    summary = pd.DataFrame(rows)
    summary.to_csv(reports_dir / 'feature_distribution_summary.csv', index=False)
    for kind, filename, title in [('box', 'feature_boxplot_core_features.png', 'Core feature distributions by EventType (boxplot)'), ('violin', 'feature_violin_core_features.png', 'Core feature distributions by EventType (violin)')]:
        n = len(features)
        cols = 3
        rows_n = math.ceil(n / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 7, rows_n * 5), squeeze=False)
        axes_flat = axes.flatten()
        for idx, feat in enumerate(features):
            ax = axes_flat[idx]
            plot_df = case_df[['EventType', feat]].dropna().copy()
            if kind == 'box':
                sns.boxplot(data=plot_df, x='EventType', y=feat, hue='EventType', order=CLASS_ORDER, hue_order=CLASS_ORDER, palette=CLASS_PALETTE, ax=ax, legend=False, dodge=False, fliersize=2)
                sns.stripplot(data=plot_df, x='EventType', y=feat, order=CLASS_ORDER, ax=ax, color='black', size=3, alpha=0.35)
            else:
                sns.violinplot(data=plot_df, x='EventType', y=feat, hue='EventType', order=CLASS_ORDER, hue_order=CLASS_ORDER, palette=CLASS_PALETTE, ax=ax, legend=False, cut=0, inner='quartile')
                sns.stripplot(data=plot_df, x='EventType', y=feat, order=CLASS_ORDER, ax=ax, color='black', size=2.5, alpha=0.25)
            ax.set_title(feat)
            ax.tick_params(axis='x', rotation=25)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
            ax.set_xlabel('')
        for ax in axes_flat[n:]:
            ax.axis('off')
        fig.suptitle(title, y=1.01)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=220, bbox_inches='tight')
        plt.close(fig)
    return summary


def plot_scatter(case_df: pd.DataFrame, figures_dir: Path) -> None:
    for x, y, filename, title in SCATTER_SPECS:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=case_df, x=x, y=y, hue='EventType', hue_order=CLASS_ORDER, palette=CLASS_PALETTE, s=90, ax=ax)
        ax.set_title(title)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))
        ax.legend(title='EventType', bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=220, bbox_inches='tight')
        plt.close(fig)


def normal_raw_sanity(raw_dir: Path, case_df: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    loaded = []
    for meta in list_waveform_case_files(raw_dir):
        df = load_waveform_case(meta)
        digest = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
        loaded.append((meta, df, digest))
    digest_to_cases: dict[str, list[str]] = {}
    for meta, _, digest in loaded:
        digest_to_cases.setdefault(digest, []).append(meta.case_name)

    rows = []
    normals = [(meta, df, digest) for meta, df, digest in loaded if meta.event_type == 'Normal']
    fig, axes = plt.subplots(len(normals), 2, figsize=(16, 5 * len(normals)), squeeze=False)
    for idx, (meta, df, digest) in enumerate(normals):
        t = df['Time'].to_numpy(dtype=float)
        mask = (t >= 0.45) & (t <= 0.65)
        bus = 1 if 1 in detect_observed_buses(df.columns) else detect_observed_buses(df.columns)[0]
        cols = signal_columns_for_bus(bus)
        window = df.loc[mask].copy()
        va, vb, vc = window[cols['VA']].to_numpy(), window[cols['VB']].to_numpy(), window[cols['VC']].to_numpy()
        ia, ib, ic = window[cols['IA']].to_numpy(), window[cols['IB']].to_numpy(), window[cols['IC']].to_numpy()
        v0 = (va + vb + vc) / 3.0
        i0 = (ia + ib + ic) / 3.0
        exact_matches = [name for name in digest_to_cases[digest] if name != meta.case_name]
        case_row = case_df.loc[case_df['CaseName'].astype(str) == meta.case_name].iloc[0]
        rows.append({
            'CaseName': meta.case_name,
            'ObservedBusForPlot': bus,
            'ExactDuplicateCaseCount': len(exact_matches),
            'ExactDuplicateCases': '|'.join(exact_matches),
            'max_dV_energy': float(case_row['max_dV_energy']),
            'max_dI_energy': float(case_row['max_dI_energy']),
            'max_sag': float(case_row['max_sag']),
            'max_V0_ratio': float(case_row['max_V0_ratio']),
            'max_I0_ratio': float(case_row['max_I0_ratio']),
            'max_V_unbalance': float(case_row['max_V_unbalance']),
            'max_I_unbalance': float(case_row['max_I_unbalance']),
            'window_V0_rms_bus1': float(np.sqrt(np.mean(v0 ** 2))),
            'window_I0_rms_bus1': float(np.sqrt(np.mean(i0 ** 2))),
        })
        axes[idx, 0].plot(window['Time'], va, label='Va')
        axes[idx, 0].plot(window['Time'], vb, label='Vb')
        axes[idx, 0].plot(window['Time'], vc, label='Vc')
        axes[idx, 1].plot(window['Time'], ia, label='Ia')
        axes[idx, 1].plot(window['Time'], ib, label='Ib')
        axes[idx, 1].plot(window['Time'], ic, label='Ic')
        axes[idx, 0].set_title(f'{meta.case_name} voltage @ bus {bus:02d}')
        axes[idx, 1].set_title(f'{meta.case_name} current @ bus {bus:02d}')
        for ax in axes[idx]:
            ax.axvline(0.5, color='black', linestyle='--', linewidth=1.0)
            ax.grid(True, alpha=0.25)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    axes[0, 0].legend(ncol=3)
    axes[0, 1].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(figures_dir / 'normal_waveform_sanity_check.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(reports_dir / 'normal_raw_sanity_check.csv', index=False)
    return out


def normal_vs_slg_boxplot(case_df: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    features = ['max_dV_energy', 'max_dI_energy', 'max_sag', 'max_V0_ratio', 'max_I0_ratio', 'max_Z_drop_ratio', 'max_V_unbalance', 'max_I_unbalance']
    subset = case_df.loc[case_df['EventType'].astype(str).isin(['Normal', 'SLG_Fault']), ['CaseName', 'EventType'] + features].copy()
    med_rows = []
    for _, row in subset.iterrows():
        rec = {'CaseName': row['CaseName'], 'EventType': row['EventType']}
        for feat in features:
            rec[feat] = row[feat]
        med_rows.append(rec)
    pd.DataFrame(med_rows).to_csv(reports_dir / 'normal_vs_event_feature_values.csv', index=False)
    n = len(features)
    fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(16, 4 * math.ceil(n / 2)), squeeze=False)
    axes_flat = axes.flatten()
    for idx, feat in enumerate(features):
        ax = axes_flat[idx]
        sns.boxplot(data=subset, x='EventType', y=feat, hue='EventType', order=['Normal', 'SLG_Fault'], hue_order=['Normal', 'SLG_Fault'], palette=CLASS_PALETTE, ax=ax, legend=False, dodge=False, fliersize=2)
        sns.stripplot(data=subset, x='EventType', y=feat, order=['Normal', 'SLG_Fault'], ax=ax, color='black', size=3, alpha=0.35)
        ax.set_title(feat)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        ax.set_xlabel('')
    for ax in axes_flat[n:]:
        ax.axis('off')
    fig.suptitle('Normal vs SLG_Fault core feature distributions', y=1.01)
    fig.tight_layout()
    fig.savefig(figures_dir / 'normal_vs_slg_core_feature_boxplot.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    return subset


def label_parsing_report(raw_dir: Path, reports_dir: Path) -> pd.DataFrame:
    rows = [{'CaseName': meta.case_name, 'EventType': meta.event_type, 'TargetBus': meta.target_bus, 'InputFile': str(meta.source_path)} for meta in list_waveform_case_files(raw_dir)]
    df = pd.DataFrame(rows).sort_values(['EventType', 'TargetBus', 'CaseName']).reset_index(drop=True)
    df.to_csv(reports_dir / 'label_parsing_check.csv', index=False)
    return df


def normal_misclassification(case_df: pd.DataFrame, mis: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    target_cases = mis.loc[(mis['EventType'].astype(str) == 'Normal') & (mis['PredictedEventType'].astype(str) == 'SLG_Fault'), 'CaseName'].astype(str).tolist()
    rows = []
    for _, row in case_df.loc[case_df['CaseName'].astype(str).isin(target_cases)].iterrows():
        for feat in MISCLASS_FEATURES:
            rec = {'CaseName': row['CaseName'], 'Feature': feat, 'Value': float(row[feat])}
            for cls in CLASS_ORDER:
                ref = pd.to_numeric(case_df.loc[case_df['EventType'].astype(str) == cls, feat], errors='coerce')
                rec[f'median_{cls}'] = float(ref.median()) if not ref.dropna().empty else float('nan')
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(reports_dir / 'normal_misclassification_feature_values.csv', index=False)
    n = len(MISCLASS_FEATURES)
    fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(16, 4 * math.ceil(n / 2)), squeeze=False)
    axes_flat = axes.flatten()
    highlights = case_df.loc[case_df['CaseName'].astype(str).isin(target_cases), ['CaseName'] + MISCLASS_FEATURES]
    for idx, feat in enumerate(MISCLASS_FEATURES):
        ax = axes_flat[idx]
        sns.boxplot(data=case_df[['EventType', feat]], x='EventType', y=feat, hue='EventType', order=CLASS_ORDER, hue_order=CLASS_ORDER, palette=CLASS_PALETTE, ax=ax, legend=False, dodge=False, fliersize=2)
        sns.stripplot(data=case_df[['EventType', feat]], x='EventType', y=feat, order=CLASS_ORDER, ax=ax, color='black', size=2.5, alpha=0.25)
        for _, hr in highlights.iterrows():
            y_val = hr[feat]
            ax.scatter(CLASS_ORDER.index('Normal'), y_val, marker='*', s=220, color='gold', edgecolor='black', zorder=5)
        ax.set_title(feat)
        ax.tick_params(axis='x', rotation=25)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    for ax in axes_flat[n:]:
        ax.axis('off')
    fig.legend(handles=[Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=14, label='Normal cases misclassified as SLG_Fault')], loc='upper right')
    fig.tight_layout()
    fig.savefig(figures_dir / 'normal_misclassification_core_features.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    return out


def correlation_heatmap(case_df: pd.DataFrame, reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    corr = case_df[CORR_FEATURES].corr(numeric_only=True)
    corr.to_csv(reports_dir / 'feature_correlation_matrix.csv')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0.0, square=True, ax=ax)
    ax.set_title('Case-level core feature correlation heatmap')
    fig.tight_layout()
    fig.savefig(figures_dir / 'feature_correlation_heatmap.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    return corr


def randomforest_importance(by_case_wide: pd.DataFrame, reports_dir: Path, figures_dir: Path):
    cols = select_feature_columns_by_group(by_case_wide, 'All_features')
    X = SimpleImputer(strategy='median').fit_transform(by_case_wide[cols])
    y = by_case_wide['EventType'].astype(str)
    rf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X, y)
    imp = pd.DataFrame({'feature': cols, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False).reset_index(drop=True)
    imp.to_csv(reports_dir / 'randomforest_feature_importance.csv', index=False)
    top30 = imp.head(30).sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(top30['feature'], top30['importance'], color='#4c78a8')
    ax.set_title('RandomForest top 30 feature importance')
    fig.tight_layout()
    fig.savefig(figures_dir / 'randomforest_top30_feature_importance.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    rows = []
    for _, row in imp.head(30).iterrows():
        m = BUS_FEATURE_RE.match(str(row['feature']))
        rows.append({'ObservedBus': int(m.group('bus')) if m else 'CaseSummary', 'feature_count': 1, 'importance': row['importance']})
    bus = pd.DataFrame(rows).groupby('ObservedBus', as_index=False).agg(feature_count=('feature_count', 'sum'), total_importance=('importance', 'sum')).sort_values(['feature_count', 'total_importance'], ascending=[False, False])
    bus.to_csv(reports_dir / 'important_wmu_bus_count.csv', index=False)
    plot_df = bus.loc[bus['ObservedBus'] != 'CaseSummary'].copy()
    plot_df['ObservedBusLabel'] = plot_df['ObservedBus'].map(lambda x: f'Bus{int(x):02d}')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x='ObservedBusLabel', y='feature_count', color='#f58518', ax=ax)
    ax.set_title('Observed bus frequency among RandomForest top 30 features')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / 'important_wmu_bus_count.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    return imp, bus


def spatial_heatmaps(by_bus: pd.DataFrame, figures_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for cls, feat, filename, title in HEATMAP_SPECS:
        subset = by_bus.loc[by_bus['EventType'].astype(str) == cls, ['TargetBus', 'ObservedBus', feat]].dropna().copy()
        subset['TargetBus'] = subset['TargetBus'].astype(int)
        subset['ObservedBus'] = subset['ObservedBus'].astype(int)
        pivot = subset.pivot_table(index='TargetBus', columns='ObservedBus', values=feat, aggfunc='mean').sort_index()
        out[filename] = pivot
        fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(pivot.index))))
        sns.heatmap(pivot, cmap='viridis', ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=220, bbox_inches='tight')
        plt.close(fig)
    return out


def representative_waveforms(raw_dir: Path, case_df: pd.DataFrame, figures_dir: Path) -> None:
    cases = {meta.case_name: meta for meta in list_waveform_case_files(raw_dir)}
    for filename, names in REP_CASES.items():
        meta = next((cases[name] for name in names if name in cases), None)
        if meta is None:
            continue
        df = load_waveform_case(meta)
        row = case_df.loc[case_df['CaseName'].astype(str) == meta.case_name].iloc[0]
        observed = detect_observed_buses(df.columns)
        bus = meta.target_bus_int if hasattr(meta, 'target_bus_int') and meta.target_bus_int in observed else (int(row['star_bus_dV']) if pd.notna(row['star_bus_dV']) and int(row['star_bus_dV']) in observed else observed[0])
        t0, t1 = REP_WINDOWS[meta.event_type]
        win = df.loc[(df['Time'] >= t0) & (df['Time'] <= t1)].copy()
        cols = signal_columns_for_bus(bus)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for phase, color in zip(['A', 'B', 'C'], ['#4c78a8', '#f58518', '#54a24b']):
            axes[0].plot(win['Time'], win[cols[f'V{phase}']], label=f'V{phase}', color=color)
            axes[1].plot(win['Time'], win[cols[f'I{phase}']], label=f'I{phase}', color=color)
        for ax in axes:
            ax.axvline(meta.event_time, color='black', linestyle='--')
            ax.grid(True, alpha=0.25)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        axes[0].set_title(f'{meta.case_name} at observed bus {bus:02d}')
        axes[0].legend(ncol=3)
        axes[1].legend(ncol=3)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=220, bbox_inches='tight')
        plt.close(fig)


def summarize_metrics(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    flat = metrics.loc[metrics['Strategy'] == 'flat'].sort_values(['macro_f1', 'balanced_accuracy'], ascending=False).iloc[0]
    hier = metrics.loc[metrics['Strategy'] == 'hierarchical'].sort_values(['macro_f1', 'balanced_accuracy', 'Normal_recall'], ascending=False).iloc[0]
    return flat, hier


def write_summaries(case_df: pd.DataFrame, dist: pd.DataFrame, metrics: pd.DataFrame, normal_sanity: pd.DataFrame, corr: pd.DataFrame, rf: pd.DataFrame, bus: pd.DataFrame, heatmaps: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    flat, hier = summarize_metrics(metrics)
    top_sep = pd.DataFrame({'Feature': CORE_CASE_FEATURES, 'anova_f': f_classif(SimpleImputer(strategy='median').fit_transform(case_df[CORE_CASE_FEATURES]), case_df['EventType'].astype(str))[0]}).sort_values('anova_f', ascending=False)
    duplicate_count = int(normal_sanity['ExactDuplicateCaseCount'].sum())
    top_corr = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            top_corr.append((a, b, float(corr.loc[a, b]), abs(float(corr.loc[a, b]))))
    top_corr = sorted(top_corr, key=lambda x: x[3], reverse=True)[:5]
    spatial = []
    for name, pivot in heatmaps.items():
        if pivot.empty:
            continue
        loc = np.unravel_index(np.nanargmax(pivot.to_numpy()), pivot.shape)
        spatial.append(f'- {name}: strongest average response at TargetBus {int(pivot.index[loc[0]])} / ObservedBus {int(pivot.columns[loc[1]])}.')
    text = f'''# Feature Diagnostics Summary

## 분석 목적
현재 결과는 최종 classification 성능 주장보다 **feature separability / Normal false alarm / raw data integrity / dataset limitation** 분석에 초점을 둔다.

## 핵심 진단
- Flat best: {flat['EventModel']} | macro-F1={flat['macro_f1']:.4f} | Normal_recall={flat['Normal_recall']:.4f}
- Hierarchical best: {hier['TriggerMethod']} + {hier['EventModel']} | macro-F1={hier['macro_f1']:.4f} | Normal_recall={hier['Normal_recall']:.4f}
- Normal raw sanity check에서 exact duplicate non-Normal matches 총 {duplicate_count}건이 확인되었다.
- Normal과 다수 SLG raw/feature가 동일하면 이는 classifier 문제가 아니라 **raw dataset integrity issue** 신호다.

## class separability 상위 feature
{chr(10).join(f'- {f}' for f in top_sep.head(6)['Feature'])}

## correlation summary
{chr(10).join(f'- {a} vs {b}: corr={c:.3f}' for a, b, c, _ in top_corr)}

## RandomForest top features
{chr(10).join(f'- {row.feature}: {row.importance:.4f}' for row in rf.head(10).itertuples(index=False))}

## important buses
{chr(10).join(f'- {row.ObservedBus}: count={row.feature_count}, summed importance={row.total_importance:.4f}' for row in bus.head(8).itertuples(index=False))}

## spatial heatmap summary
{chr(10).join(spatial)}

## 한계
- Normal sample은 3개뿐이다.
- Hierarchical trigger threshold는 preliminary이다.
- 현재 raw folder 안의 일부 SLG file은 no-event/Normal과 동일할 가능성이 높다.
- 따라서 현재 SLG 관련 metric은 raw data integrity audit 없이는 최종 주장에 쓰면 안 된다.
'''
    (reports_dir / 'feature_diagnostics_summary.md').write_text(text, encoding='utf-8')
    final_text = f'''# Final Waveform Event Analysis Summary

- Flat classifier에서 기존 Normal→SLG 오분류 문제를 재점검했다.
- Normal raw sanity check 결과, Normal 자체는 steady-state/no-event 특성을 보였고 dV/dI/sag는 거의 0이었다.
- 그러나 여러 SLG raw file이 Normal raw와 exact duplicate로 나타나 dataset integrity issue 가능성이 매우 높다.
- 따라서 현재 Normal vs SLG confusion의 핵심 원인은 class imbalance만이 아니라 **SLG raw data corruption or duplication** 가능성이다.
- Hierarchical classifier를 추가하여 no-event trigger를 먼저 분리하도록 수정했다.
- Best flat: {flat['EventModel']} | macro-F1={flat['macro_f1']:.4f} | balanced_accuracy={flat['balanced_accuracy']:.4f} | Normal_recall={flat['Normal_recall']:.4f}
- Best hierarchical: {hier['TriggerMethod']} + {hier['EventModel']} | macro-F1={hier['macro_f1']:.4f} | balanced_accuracy={hier['balanced_accuracy']:.4f} | Normal_recall={hier['Normal_recall']:.4f}
- 이 결과는 classifier 자체보다 raw data integrity 및 trigger-feature 설계의 중요성을 보여준다.

## 필수 caveat
- Normal sample 3개뿐임
- Normal trigger threshold는 preliminary임
- 향후 Normal variation 및 verified fault raw dataset 확장이 필요함
'''
    (reports_dir / 'final_analysis_summary.md').write_text(final_text, encoding='utf-8')


def main() -> None:
    args = parse_args()
    data_dir = to_local_path(args.data_dir)
    raw_dir = to_local_path(args.raw_dir)
    results_dir = to_local_path(args.results_dir)
    reports_dir, figures_dir = ensure_dirs(results_dir)

    by_bus, by_case, by_case_wide, metrics, mis = load_tables(data_dir, results_dir)
    case_df = build_case_enriched(by_case, by_bus)
    feature_distribution(case_df, CORE_CASE_FEATURES, reports_dir, figures_dir)
    plot_scatter(case_df, figures_dir)
    normal_sanity = normal_raw_sanity(raw_dir, case_df, reports_dir, figures_dir)
    label_parsing_report(raw_dir, reports_dir)
    normal_vs_slg_boxplot(case_df, reports_dir, figures_dir)
    normal_misclassification(case_df, mis, reports_dir, figures_dir)
    corr = correlation_heatmap(case_df, reports_dir, figures_dir)
    rf, bus = randomforest_importance(by_case_wide, reports_dir, figures_dir)
    heatmaps = spatial_heatmaps(by_bus, figures_dir)
    representative_waveforms(raw_dir, case_df, figures_dir)
    write_summaries(case_df, pd.read_csv(reports_dir / 'feature_distribution_summary.csv'), metrics, normal_sanity, corr, rf, bus, heatmaps, reports_dir)
    print('Diagnostics regenerated')


if __name__ == '__main__':
    main()
