# Waveform Event Analysis Results

This folder contains the compact, git-trackable outputs from the corrected 83-file 3-phase waveform event analysis pipeline.

## Dataset basis for the current tracked results

- Total raw event files: 83
- Normal: 3
- LoadSwitch: 20
- SLG_Fault: 30
- ThreePhase_Fault: 30

The LoadSwitch cases correspond to the regenerated **15% abrupt load increase** dataset at load buses only.

## Reports

Stored in `reports/`:
- `data_quality_summary.md` - dataset quality snapshot
- `classification_full_wmu_metrics.csv` - full-WMU LOO classification metrics
- `classification_full_wmu_report.txt` - narrative classification report
- `confusion_matrix_full_wmu.csv` - confusion-matrix counts for the best full-WMU model
- `misclassified_cases_full_wmu.csv` - case-level misclassifications for the best full-WMU model
- `feature_ablation_metrics.csv` - feature-group comparison table
- `feature_ablation_used_columns.txt` - actual feature columns/counts per ablation group
- `sensor_count_curve.csv` - limited-WMU performance curves
- `selected_wmu_by_k.csv` - greedy WMU order by sensor count
- `fault_localization_preliminary.csv` - self-matching and strict-LOO localization summary
- `fault_localization_debug.csv` - per-case localization debug table
- `fault_localization_notes.txt` - localization caveats
- `final_analysis_summary.md` - one-page summary of the current run

## Figures

Stored in `figures/`:
- `confusion_matrix_full_wmu.png`
- `feature_ablation_macro_f1.png`
- `feature_ablation_loadswitch_recall.png`
- `sensor_count_macro_f1.png`
- `sensor_count_balanced_accuracy.png`
- `sensor_count_loadswitch_recall.png`
- `sensor_count_fault_precision.png`
- `fault_localization_confusion_matrix.png`

## Not tracked here

The following remain outside git because of size:
- raw xlsx/csv waveform files
- the full external output directory
- large intermediate tables such as full feature tables
- `sensor_selection_debug.csv` from the external output directory

Regenerate the external outputs with:

```bash
python scripts/run_waveform_event_analysis.py \
  --input-dir "C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw" \
  --output-dir "C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_data" \
  --f0 50
```
