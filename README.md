# WMU Project

Python tooling for WMU analysis on the IEEE 30-bus system.

## Existing workflow: SLG observability analysis

The original repository workflow is preserved for the IEEE 30-bus SLG fault sweep observability study.

Key entry points:
- `src/wmu_project/analysis.py`
- `scripts/analyze_observability.py`
- `scripts/figure1_coverage_vs_num_wmu.py` ... `scripts/figure6_3d_scatter.py`
- `docs/research_summary.md`

## New workflow: Waveform Event Classification Workflow

This repository now also includes a separate 3-phase waveform event analysis pipeline for:
- `Normal`
- `LoadSwitch`
- `SLG_Fault`
- `ThreePhase_Fault`

Research goal:
- extract bus-level WMU waveform features from raw 3-phase event files,
- evaluate multi-class event classification,
- compare feature ablations,
- study limited-WMU sensor-count performance,
- and run preliminary fault-localization separability analysis.

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

### Input dataset

Primary raw-data directory used in this run:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw`
- WSL/Linux path used here: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_raw`

Output directory used in this run:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_data`
- WSL/Linux path used here: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_data`

Raw Excel files are intentionally ignored by git.

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
- `reports/feature_ablation_metrics.csv`
- `reports/sensor_count_curve.csv`
- `reports/fault_localization_preliminary.csv`
- `reports/final_analysis_summary.md`
- `figures/*.png`

Tracked repo summaries:
- `results/waveform_event_analysis/reports/`
- `results/waveform_event_analysis/figures/`

### Current run snapshot

From the 2026-05-20 run on the available dataset:
- input xlsx files: `93`
- class counts: `Normal=3`, `LoadSwitch=30`, `SLG_Fault=30`, `ThreePhase_Fault=30`
- data-quality status: `OK=93`, `WARNING=0`, `FAILED=0`
- best full-WMU LOO classifier: `RandomForest`
- best full-WMU macro-F1: `0.7381`
- best balanced accuracy: `0.7500`

See `results/waveform_event_analysis/reports/final_analysis_summary.md` for the detailed summary.

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

Small summary CSVs and key figures under `results/waveform_event_analysis/` remain trackable.
