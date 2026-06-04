# IBR-background WMU waveform analysis

This directory stores compact, git-trackable summary artifacts for the modified IEEE 30-bus WMU waveform dataset where an IBR-like 25 Hz SSO condition is present in every case.

Raw and large intermediate files are intentionally kept outside the repository:

- Raw CSV input: `C:\Users\user\Documents\MATLAB\WMU_final\WMU_batch_raw_ibr_background`
- Analysis output: `C:\Users\user\Documents\MATLAB\WMU_final\WMU_batch_data_ibr_background`

## Dataset inventory

- `SSO_Normal`: 3 cases
- `SSO_LoadSwitch`: 21 cases
- `SSO_SLG_Fault`: 30 cases
- `SSO_ThreePhase_Fault`: 30 cases
- Total: 84 CSV files

All cases include the IBR-like SSO background (`f = 25 Hz`, active window `0.02 s` to `0.48 s`). LoadSwitch cases use the working model's `LoadAdd<bus>` blocks set to 15% of the existing `Load<bus>` P/QL values before enabling the corresponding breaker.

## Key generated reports

- `reports/final_analysis_summary.md`
- `reports/data_quality_summary.md`
- `reports/classification_flat_vs_hierarchical_metrics.csv`
- `reports/feature_ablation_metrics.csv`
- `reports/sensor_count_curve.csv`
- `reports/fault_localization_preliminary.csv`
- `reports/dataset_case_index.csv`

## Key figures

- `figures/confusion_matrix_full_wmu.png`
- `figures/confusion_matrix_hierarchical_full_wmu.png`
- `figures/feature_ablation_macro_f1.png`
- `figures/sensor_count_macro_f1.png`
- `figures/fault_localization_confusion_matrix.png`

## Snapshot

The 2026-06-03 run processed 84 cases with data quality `OK=84`, `WARNING=0`, `FAILED=0`. The best full-WMU RandomForest and hierarchical RandomForest runs reached macro-F1 `1.0000` on leave-one-out evaluation. Treat this as an internal dataset/feature diagnostic result because the dataset has one fixed SSO background condition and limited repetitions.
