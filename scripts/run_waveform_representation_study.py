from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wmu_project.waveform_utils import EVENT_CLASS_ORDER, FAULT_EVENTS
from wmu_project.waveform_classification import aggregate_trigger_features, apply_rule_trigger, build_rule_thresholds

sns.set_theme(style='whitegrid', context='talk')
CLASSES = ['Normal', 'LoadSwitch', 'SLG_Fault', 'ThreePhase_Fault']
FAULT_CLASSES = ['SLG_Fault', 'ThreePhase_Fault']
PALETTE = {'Normal':'#4c78a8','LoadSwitch':'#f58518','SLG_Fault':'#e45756','ThreePhase_Fault':'#72b7b2'}


def load_tables():
    base = Path('/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_data')
    by_bus = pd.read_csv(base / 'feature_table_by_bus.csv')
    by_case = pd.read_csv(base / 'feature_table_by_case.csv')
    by_case_wide = pd.read_csv(base / 'feature_table_by_case_wide.csv')
    return by_bus, by_case, by_case_wide


def ensure_dirs():
    rep = ROOT / 'results' / 'waveform_event_analysis' / 'reports'
    fig = ROOT / 'results' / 'waveform_event_analysis' / 'figures'
    feat = ROOT / 'features'
    rep.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    feat.mkdir(parents=True, exist_ok=True)
    return rep, fig, feat


def compute_event_type_summary(by_bus: pd.DataFrame, by_case: pd.DataFrame) -> pd.DataFrame:
    df = by_bus.copy()
    eps = 1e-12
    sag_bc_mean = df[['sag_B','sag_C']].mean(axis=1)
    dv_bc_mean = df[['dV_energy_B','dV_energy_C']].mean(axis=1)
    di_bc_mean = df[['dI_energy_B','dI_energy_C']].mean(axis=1)
    df['phase_sag_std'] = df[['sag_A','sag_B','sag_C']].std(axis=1)
    df['phase_sag_imbalance'] = df['phase_sag_std'] / (df[['sag_A','sag_B','sag_C']].mean(axis=1) + eps)
    df['dV_phase_imbalance'] = df[['dV_energy_A','dV_energy_B','dV_energy_C']].std(axis=1) / (df[['dV_energy_A','dV_energy_B','dV_energy_C']].mean(axis=1) + eps)
    df['dI_phase_imbalance'] = df[['dI_energy_A','dI_energy_B','dI_energy_C']].std(axis=1) / (df[['dI_energy_A','dI_energy_B','dI_energy_C']].mean(axis=1) + eps)
    df['A_phase_sag_dominance'] = df['sag_A'] / (sag_bc_mean + eps)
    df['A_phase_dV_dominance'] = df['dV_energy_A'] / (dv_bc_mean + eps)
    df['A_phase_dI_dominance'] = df['dI_energy_A'] / (di_bc_mean + eps)
    df['sag_A_minus_mean_BC'] = df['sag_A'] - sag_bc_mean
    df['sag_A_over_mean_BC'] = df['sag_A'] / (sag_bc_mean + eps)
    df['dV_A_minus_mean_BC'] = df['dV_energy_A'] - dv_bc_mean
    df['dV_A_over_mean_BC'] = df['dV_energy_A'] / (dv_bc_mean + eps)
    df['dI_A_minus_mean_BC'] = df['dI_energy_A'] - di_bc_mean
    df['dI_A_over_mean_BC'] = df['dI_energy_A'] / (di_bc_mean + eps)
    df['phase_sag_symmetry_score'] = 1.0 / (1.0 + df['phase_sag_std'])
    df['threephase_balance_score'] = 1.0 / (1.0 + df['phase_sag_imbalance'] + df['dV_phase_imbalance'] + df['dI_phase_imbalance'])
    df['V_unbalance_delta'] = df['Delta_V_unbalance']
    df['I_unbalance_delta'] = df['Delta_I_unbalance']
    df['zero_sequence_dominance_score'] = df['V0_ratio'] + df['I0_ratio']
    df['negative_sequence_dominance_score'] = df['V2_ratio'] + df['I2_ratio']
    df['zero_to_negative_combo'] = (df['V0_ratio'] + df['I0_ratio']) / (df['V2_ratio'] + df['I2_ratio'] + eps)
    df['SLG_score_candidate'] = df['A_phase_sag_dominance'] + df['A_phase_dV_dominance'] + df['A_phase_dI_dominance'] + df['zero_sequence_dominance_score'] + df['V_unbalance_delta'] + df['I_unbalance_delta']
    df['ThreePhase_score_candidate'] = df[['sag_A','sag_B','sag_C']].mean(axis=1) + df['threephase_balance_score'] - df['phase_sag_imbalance']

    summary_cols = [
        'phase_sag_std','phase_sag_imbalance','dV_phase_imbalance','dI_phase_imbalance',
        'A_phase_sag_dominance','A_phase_dV_dominance','A_phase_dI_dominance',
        'sag_A_minus_mean_BC','sag_A_over_mean_BC','dV_A_minus_mean_BC','dV_A_over_mean_BC','dI_A_minus_mean_BC','dI_A_over_mean_BC',
        'phase_sag_symmetry_score','threephase_balance_score','V_unbalance_delta','I_unbalance_delta',
        'V0_ratio','I0_ratio','V2_ratio','I2_ratio','zero_sequence_dominance_score','negative_sequence_dominance_score','zero_to_negative_combo',
        'Z_drop_ratio','dV_HF_ratio_3ph_max','dV_Res_ratio_3ph_max','dI_HF_ratio_3ph_max','dI_Res_ratio_3ph_max','liss_corr_abs_min','liss_area_norm_min',
        'SLG_score_candidate','ThreePhase_score_candidate'
    ]

    rows = []
    group_cols = ['CaseName','EventType','TargetBus']
    for keys, g in df.groupby(group_cols, dropna=False):
        row = {'CaseName':keys[0],'EventType':keys[1],'TargetBus':keys[2]}
        for col in summary_cols:
            s = pd.to_numeric(g[col], errors='coerce')
            row[f'{col}_max'] = float(s.max())
            row[f'{col}_mean'] = float(s.mean())
            row[f'{col}_median'] = float(s.median())
            row[f'{col}_std'] = float(s.std(ddof=0))
            row[f'{col}_q75'] = float(s.quantile(0.75))
            row[f'{col}_q95'] = float(s.quantile(0.95))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(['EventType','TargetBus','CaseName']).reset_index(drop=True)
    case_keep = ['CaseName','EventType','TargetBus','max_dV_energy','max_dI_energy','max_sag','max_HF_ratio','max_Res_ratio','max_V0_ratio','max_I0_ratio','max_V_unbalance','max_I_unbalance','max_Z_drop_ratio']
    if not by_case.empty and all(col in by_case.columns for col in case_keep):
        summary = summary.merge(by_case[case_keep], on=['CaseName','EventType','TargetBus'], how='left')
    return summary


def feature_sets(event_summary: pd.DataFrame, by_case_wide: pd.DataFrame):
    existing = [c for c in by_case_wide.columns if c not in ['CaseName','EventType','TargetBus','EventTime','SamplingRateHz','ObservedBusCount']]
    event_summary_cols = [c for c in event_summary.columns if c not in ['CaseName','EventType','TargetBus']]
    hybrid = event_summary_cols + [c for c in ['max_dV_energy','max_dI_energy','max_sag','max_HF_ratio','max_Res_ratio','max_V0_ratio','max_I0_ratio','max_V_unbalance','max_I_unbalance','max_Z_drop_ratio'] if c in event_summary.columns]
    return {'existing_wide_features': existing, 'event_type_summary_features': event_summary_cols, 'hybrid_features': sorted(set(hybrid))}


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imp.fit_transform(train_df[feature_cols]))
    Xte = scaler.transform(imp.transform(test_df[feature_cols]))
    return Xtr, Xte


def loo_eval(df: pd.DataFrame, feature_cols: list[str], grouped_by_targetbus: bool = False, binary_fault: bool = False):
    labels = FAULT_CLASSES if binary_fault else CLASSES
    rows = []
    preds = []
    model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced', n_jobs=-1)
    work = df.copy().reset_index(drop=True)
    if binary_fault:
        work = work[work['EventType'].isin(FAULT_CLASSES)].reset_index(drop=True)
    for i in range(len(work)):
        test = work.iloc[[i]]
        if grouped_by_targetbus and pd.notna(test.iloc[0]['TargetBus']) and test.iloc[0]['EventType'] in FAULT_CLASSES:
            train = work.drop(index=i)
            train = train[~((train['EventType'].isin(FAULT_CLASSES)) & (train['TargetBus'] == test.iloc[0]['TargetBus']))]
        else:
            train = work.drop(index=i)
        Xtr, Xte = preprocess(train, test, feature_cols)
        ytr = train['EventType'].astype(str).to_numpy()
        yte = test['EventType'].astype(str).to_numpy()
        fitted = clone(model).fit(Xtr, ytr)
        pred = fitted.predict(Xte)[0]
        preds.append(pred)
        row = test[['CaseName','EventType','TargetBus']].iloc[0].to_dict()
        row['y_true'] = yte[0]
        row['y_pred'] = pred
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    y_true = pred_df['y_true'].to_numpy()
    y_pred = pred_df['y_pred'].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metric = {'accuracy': accuracy_score(y_true, y_pred), 'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0), 'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)}
    for i, lab in enumerate(labels):
        metric[f'recall_{lab}'] = recall[i]
        metric[f'precision_{lab}'] = precision[i]
        metric[f'f1_{lab}'] = f1[i]
    if not binary_fault:
        metric['Fault_precision'] = float(np.mean(np.isin(y_pred[np.isin(y_true, FAULT_CLASSES)], FAULT_CLASSES))) if np.any(np.isin(y_true, FAULT_CLASSES)) else np.nan
        metric['swap_rate'] = float(np.mean(((y_true == 'SLG_Fault') & (y_pred == 'ThreePhase_Fault')) | ((y_true == 'ThreePhase_Fault') & (y_pred == 'SLG_Fault'))))
    else:
        metric['swap_rate'] = float(np.mean(y_true != y_pred))
    return metric, pred_df


def plot_confusion(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)


def fault_pair_distance_debug(by_case_wide: pd.DataFrame, rep: Path, fig: Path):
    fault = by_case_wide[by_case_wide['EventType'].isin(FAULT_CLASSES)].reset_index(drop=True)
    feature_cols = [c for c in by_case_wide.columns if c not in ['CaseName','EventType','TargetBus','EventTime','SamplingRateHz','ObservedBusCount']]
    X = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(fault[feature_cols]))
    rows=[]
    for i,row in fault.iterrows():
        d = np.linalg.norm(X - X[i], axis=1)
        same_bus_opp = fault[(fault['TargetBus']==row['TargetBus']) & (fault['EventType']!=row['EventType'])].index
        same_type_other = fault[(fault['EventType']==row['EventType']) & (fault['TargetBus']!=row['TargetBus'])].index
        opp_type_other = fault[(fault['EventType']!=row['EventType']) & (fault['TargetBus']!=row['TargetBus'])].index
        rows.append({'CaseName':row['CaseName'],'EventType':row['EventType'],'TargetBus':row['TargetBus'],'distance_opposite_fault_same_bus':float(d[same_bus_opp[0]]) if len(same_bus_opp) else np.nan,'min_distance_same_type_other_bus':float(d[same_type_other].min()) if len(same_type_other) else np.nan,'min_distance_opposite_type_other_bus':float(d[opp_type_other].min()) if len(opp_type_other) else np.nan,'same_bus_opposite_is_nearest': bool(len(same_bus_opp) and d[same_bus_opp[0]] <= min(d[same_type_other].min() if len(same_type_other) else np.inf, d[opp_type_other].min() if len(opp_type_other) else np.inf))})
    out=pd.DataFrame(rows)
    out.to_csv(rep/'fault_pair_distance_debug.csv', index=False)
    melt=out.melt(id_vars=['CaseName','EventType','TargetBus'], value_vars=['distance_opposite_fault_same_bus','min_distance_same_type_other_bus','min_distance_opposite_type_other_bus'], var_name='DistanceType', value_name='Distance')
    fig_,ax=plt.subplots(figsize=(10,6)); sns.boxplot(data=melt,x='DistanceType',y='Distance',hue='EventType',palette=PALETTE,ax=ax); ax.tick_params(axis='x',rotation=20); fig_.tight_layout(); fig_.savefig(fig/'fault_pair_distance_boxplot.png',dpi=220,bbox_inches='tight'); plt.close(fig_)
    return out


def grouped_and_representation_study(event_summary: pd.DataFrame, by_case_wide: pd.DataFrame, rep: Path, fig: Path):
    sets = feature_sets(event_summary, by_case_wide)
    comp_rows=[]
    grouped_rows=[]
    subtype_rows=[]
    pred_event_summary=None
    for name, cols in sets.items():
        source = by_case_wide if name=='existing_wide_features' else event_summary
        for grouped in [False, True]:
            metric, pred = loo_eval(source, cols, grouped_by_targetbus=grouped, binary_fault=False)
            comp_rows.append({'FeatureSetting':name,'Evaluation':'GroupedByTargetBus' if grouped else 'LOO', **metric})
            if grouped:
                grouped_rows.append({'FeatureSetting':name, **metric})
            if name=='event_type_summary_features' and not grouped:
                pred_event_summary = pred.copy()
            bmetric, bpred = loo_eval(source, cols, grouped_by_targetbus=grouped, binary_fault=True)
            subtype_rows.append({'FeatureSetting':name,'Evaluation':'GroupedByTargetBus' if grouped else 'LOO', **bmetric})
            if name=='event_type_summary_features' and not grouped:
                plot_confusion(pred['y_true'], pred['y_pred'], CLASSES, fig/'confusion_matrix_event_type_summary.png', 'Event-type summary features confusion matrix')
                plot_confusion(bpred['y_true'], bpred['y_pred'], FAULT_CLASSES, fig/'confusion_matrix_fault_subtype_binary.png', 'Fault subtype binary confusion matrix')
        if pred_event_summary is not None:
            plot_confusion(pred_event_summary['y_true'], pred_event_summary['y_pred'], CLASSES, fig/'confusion_matrix_grouped_by_targetbus.png', 'Grouped-by-TargetBus confusion matrix (event_type_summary_features)')
    pd.DataFrame(comp_rows).to_csv(rep/'classification_feature_representation_comparison.csv', index=False)
    pd.DataFrame(grouped_rows).to_csv(rep/'classification_grouped_by_targetbus_metrics.csv', index=False)
    pd.DataFrame(subtype_rows).to_csv(rep/'fault_subtype_binary_metrics.csv', index=False)
    return sets


def export_feature_tables(event_summary: pd.DataFrame, by_case_wide: pd.DataFrame, feat_dir: Path):
    event_summary.to_csv(feat_dir/'feature_table_event_type_summary.csv', index=False)
    by_case_wide.to_csv(feat_dir/'feature_table_localization_wide.csv', index=False)


def subtype_feature_plots(event_summary: pd.DataFrame, rep: Path, fig: Path):
    subset = event_summary[event_summary['EventType'].isin(FAULT_CLASSES)].copy()
    cols = ['A_phase_sag_dominance_max','phase_sag_imbalance_max','V0_ratio_max_max','I0_ratio_max_max','V_unbalance_delta_max_max','I_unbalance_delta_max_max','threephase_balance_score_mean','zero_sequence_dominance_score_max']
    fixed=[]
    for c in cols:
        if c in subset.columns:
            fixed.append(c)
        elif c.replace('_max_max','_max') in subset.columns:
            fixed.append(c.replace('_max_max','_max'))
        elif c.replace('_mean','_mean') in subset.columns:
            fixed.append(c)
    melt = subset[['EventType']+fixed].melt(id_vars='EventType', var_name='Feature', value_name='Value')
    melt.to_csv(rep/'slg_threephase_feature_summary.csv', index=False)
    fig_,ax=plt.subplots(figsize=(12,6)); sns.boxplot(data=melt,x='Feature',y='Value',hue='EventType',palette=PALETTE,ax=ax); ax.tick_params(axis='x',rotation=30); fig_.tight_layout(); fig_.savefig(fig/'slg_threephase_subtype_feature_boxplot.png',dpi=220,bbox_inches='tight'); plt.close(fig_)
    return melt


def event_type_importance(event_summary: pd.DataFrame, rep: Path, fig: Path):
    cols=[c for c in event_summary.columns if c not in ['CaseName','EventType','TargetBus']]
    X=SimpleImputer(strategy='median').fit_transform(event_summary[cols]); y=event_summary['EventType'].astype(str)
    rf=RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced', n_jobs=-1).fit(X,y)
    imp=pd.DataFrame({'feature':cols,'importance':rf.feature_importances_}).sort_values('importance',ascending=False).reset_index(drop=True)
    imp.to_csv(rep/'event_type_feature_importance.csv', index=False)
    top=imp.head(30).sort_values('importance',ascending=True)
    fig_,ax=plt.subplots(figsize=(10,11)); ax.barh(top['feature'], top['importance'], color='#4c78a8'); ax.set_title('Event-type summary top 30 feature importance'); fig_.tight_layout(); fig_.savefig(fig/'event_type_top30_feature_importance.png', dpi=220, bbox_inches='tight'); plt.close(fig_)


def sensor_count_split(by_bus: pd.DataFrame, rep: Path, fig: Path):
    buses=sorted(by_bus['ObservedBus'].astype(int).unique())
    # event classification summary-based
    event_rows=[]
    loc_rows=[]
    for k in range(1, len(buses)+1):
        sel=buses[:k]
        subset=by_bus[by_bus['ObservedBus'].isin(sel)].copy()
        summary = compute_event_type_summary(subset, pd.DataFrame())
        cols=[c for c in summary.columns if c not in ['CaseName','EventType','TargetBus']]
        metric,_=loo_eval(summary, cols, grouped_by_targetbus=False, binary_fault=False)
        event_rows.append({'k':k, **metric})
        # localization strict loo exact accuracy
        wide_rows=[]
        features=[c for c in subset.columns if c not in ['CaseName','EventType','TargetBus','ObservedBus','EventTime','SamplingRateHz']]
        for (case_name,event_type,target_bus),g in subset.groupby(['CaseName','EventType','TargetBus'], dropna=False):
            row={'CaseName':case_name,'EventType':event_type,'TargetBus':target_bus}
            for bus in sel:
                r=g[g['ObservedBus']==bus]
                if r.empty:
                    for feat in features: row[f'Bus{bus:02d}__{feat}']=np.nan
                else:
                    rr=r.iloc[0]
                    for feat in features: row[f'Bus{bus:02d}__{feat}']=rr[feat]
            wide_rows.append(row)
        wide=pd.DataFrame(wide_rows)
        fault=wide[wide['EventType'].isin(FAULT_CLASSES)].reset_index(drop=True)
        fcols=[c for c in fault.columns if c not in ['CaseName','EventType','TargetBus']]
        X=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(fault[fcols]))
        buses_arr=fault['TargetBus'].astype(int).to_numpy()
        pred=[]
        for i in range(len(fault)):
            d=np.linalg.norm(X-X[i], axis=1); d[i]=np.inf; pred.append(int(buses_arr[np.argmin(d)]))
        exact=float(np.mean(buses_arr==np.array(pred)))
        loc_rows.append({'k':k,'strict_loo_exact_accuracy':exact})
    pd.DataFrame(event_rows).to_csv(rep/'sensor_count_event_classification_curve.csv', index=False)
    pd.DataFrame(loc_rows).to_csv(rep/'sensor_count_localization_curve.csv', index=False)
    fig1,ax1=plt.subplots(figsize=(9,5)); df1=pd.DataFrame(event_rows); ax1.plot(df1['k'], df1['macro_f1'], marker='o'); ax1.set_title('Sensor count event classification macro-F1'); ax1.set_xlabel('k'); ax1.set_ylabel('Macro-F1'); fig1.tight_layout(); fig1.savefig(fig/'sensor_count_event_classification_macro_f1.png', dpi=220, bbox_inches='tight'); plt.close(fig1)
    fig2,ax2=plt.subplots(figsize=(9,5)); df2=pd.DataFrame(loc_rows); ax2.plot(df2['k'], df2['strict_loo_exact_accuracy'], marker='o'); ax2.set_title('Sensor count localization strict LOO accuracy'); ax2.set_xlabel('k'); ax2.set_ylabel('Exact accuracy'); fig2.tight_layout(); fig2.savefig(fig/'sensor_count_localization_accuracy.png', dpi=220, bbox_inches='tight'); plt.close(fig2)


def main():
    by_bus, by_case, by_case_wide = load_tables()
    rep, fig, feat_dir = ensure_dirs()
    event_summary = compute_event_type_summary(by_bus, by_case)
    export_feature_tables(event_summary, by_case_wide, feat_dir)
    fault_pair_distance_debug(by_case_wide, rep, fig)
    grouped_and_representation_study(event_summary, by_case_wide, rep, fig)
    event_type_importance(event_summary, rep, fig)
    subtype_feature_plots(event_summary, rep, fig)
    sensor_count_split(by_bus, rep, fig)
    print('representation study complete')

if __name__ == '__main__':
    main()
