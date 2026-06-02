# WMU Project

Python tooling for WMU analysis on the IEEE 30-bus system.

## Existing workflow: SLG observability analysis

The original repository workflow is preserved for the IEEE 30-bus SLG fault sweep observability study.

Key entry points:
- `src/wmu_project/analysis.py`
- `scripts/analyze_observability.py`
- `scripts/figure1_coverage_vs_num_wmu.py` ... `scripts/figure6_3d_scatter.py`
- `docs/research_summary.md`

## Waveform Event Classification Workflow

This repository also includes a separate 3-phase waveform event analysis pipeline for:
- `Normal`
- `LoadSwitch`
- `SLG_Fault`
- `ThreePhase_Fault`

Research goal:
- extract WMU waveform features from raw 3-phase event files,
- evaluate four-class event classification,
- compare feature ablations,
- study limited-WMU sensor-count performance,
- and run preliminary fault-localization analysis.

### Repository layout for waveform-event analysis

- `src/wmu_project/waveform_io.py`
- `src/wmu_project/waveform_quality.py`
- `src/wmu_project/waveform_features.py`
- `src/wmu_project/waveform_classification.py`
- `src/wmu_project/waveform_sensor_selection.py`
- `src/wmu_project/waveform_localization.py`
- `src/wmu_project/waveform_utils.py`
- `scripts/inspect_waveform_dataset.py`
- `scripts/export_waveform_feature_table.py`
- `scripts/run_waveform_event_analysis.py`
- `scripts/run_waveform_classification.py`
- `scripts/run_waveform_sensor_selection.py`
- `scripts/run_waveform_localization.py`
- `docs/waveform_event_analysis.md`
- `docs/waveform_feature_definitions.md`
- `docs/waveform_dataset_notes.md`
- `results/waveform_event_analysis/README.md`

### Input dataset used in the current run

Primary raw-data directory:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw`
- WSL/Linux: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_raw`

Output directory:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_data`
- WSL/Linux: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_data`

Current raw event inventory:
- `83` total event files
- `30` `SLG_Fault_Bus*.xlsx`
- `30` `ThreePhase_Fault_Bus*.xlsx`
- `3` `Normal_Case*.xlsx`
- `20` `LoadSwitch_Bus*.csv`

The LoadSwitch cases were regenerated as **15% abrupt load increases** only at buses with existing Pd or Qd:
`2, 3, 4, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 29, 30`

The pipeline now supports mixed input formats:
- xlsx for SLG / ThreePhase / Normal
- csv for LoadSwitch

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### End-to-end waveform-event analysis

```bash
python scripts/run_waveform_event_analysis.py \
  --input-dir "C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw" \
  --output-dir "C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_data" \
  --f0 50
```

### Main generated outputs

External output directory:
- `feature_table_by_bus.csv`
- `feature_table_by_case.csv`
- `feature_table_by_case_wide.csv`
- `reports/data_quality_report.csv`
- `reports/classification_full_wmu_metrics.csv`
- `reports/confusion_matrix_full_wmu.csv`
- `reports/misclassified_cases_full_wmu.csv`
- `reports/feature_ablation_metrics.csv`
- `reports/feature_ablation_used_columns.txt`
- `reports/sensor_count_curve.csv`
- `reports/selected_wmu_by_k.csv`
- `reports/sensor_selection_debug.csv`
- `reports/fault_localization_preliminary.csv`
- `reports/fault_localization_debug.csv`
- `reports/final_analysis_summary.md`
- `figures/*.png`

Tracked repo summaries:
- `results/waveform_event_analysis/reports/`
- `results/waveform_event_analysis/figures/`

### Current run snapshot

From the 2026-05-20 corrected 83-file dataset run:
- input raw event files: `83`
- class counts: `Normal=3`, `LoadSwitch=20`, `SLG_Fault=30`, `ThreePhase_Fault=30`
- data-quality status: `OK=83`, `WARNING=0`, `FAILED=0`
- best full-WMU LOO classifier: `RandomForest`
- best full-WMU macro-F1: `0.7381`
- best balanced accuracy: `0.7500`

See `results/waveform_event_analysis/reports/final_analysis_summary.md` for details.

### Normal/SLG debugging status

A later debugging pass found that the `Normal -> SLG_Fault` failure mode cannot be interpreted as a simple model weakness alone. The current workflow now also:
- audits Normal raw waveforms directly,
- checks label parsing and feature-table integrity,
- compares the original flat classifier with a hierarchical `Normal-vs-Event` trigger-first classifier, and
- treats the present outputs as **feature/dataset diagnostics**, not as a final classification claim, when `Normal` and many `SLG_Fault` cases appear numerically indistinguishable.

Key additional outputs:
- `results/waveform_event_analysis/reports/normal_raw_sanity_check.csv`
- `results/waveform_event_analysis/reports/label_parsing_check.csv`
- `results/waveform_event_analysis/reports/classification_flat_vs_hierarchical_metrics.csv`
- `results/waveform_event_analysis/figures/confusion_matrix_hierarchical_full_wmu.png`
- `results/waveform_event_analysis/figures/normal_waveform_sanity_check.png`
- `results/waveform_event_analysis/figures/normal_vs_slg_core_feature_boxplot.png`

## Git hygiene

The repository excludes raw and large local artifacts such as:
- `*.xlsx`
- `*.mat`
- `*.slx`
- `WMU_batch_raw/`
- `WMU_batch_data/`
- `outputs/`
- `results/raw/`
- `results/intermediate/`

Raw xlsx/csv event files and large intermediate tables stay outside git. Compact summary CSVs and key figures under `results/waveform_event_analysis/` remain trackable.


## IBR-like SSO scenario status

An SPS-only IBR-like SSO inspection/scaffolding pass was added for MATLAB R2025b. The current model inspection confirms that the existing workflow can safely reset all fault/loadswitch events, but a validated **physical** 20–30 Hz SPS injection path has not yet been established without adding incompatible Simscape physical-port blocks. See `docs/waveform_ibr_sso_scenario.md`.
