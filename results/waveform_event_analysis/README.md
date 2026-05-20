# Waveform Event Analysis Results

This folder contains the compact, git-trackable outputs from the 3-phase waveform event analysis pipeline.

## Reports

Stored in `reports/`:
- `data_quality_summary.md` - dataset quality snapshot
- `classification_full_wmu_metrics.csv` - full-WMU LOO classification metrics
- `classification_full_wmu_report.txt` - narrative classification report
- `feature_ablation_metrics.csv` - feature-group comparison table
- `sensor_count_curve.csv` - limited-WMU performance curves
- `selected_wmu_by_k.csv` - greedy WMU order by sensor count
- `fault_localization_preliminary.csv` - preliminary localization metrics
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
- raw xlsx waveform files
- the full external output directory
- large intermediate tables beyond the compact summaries included here

Feature tables can be regenerated with:

```bash
python scripts/run_waveform_event_analysis.py \
  --input-dir "C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw" \
  --output-dir "C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_data" \
  --f0 50
```
