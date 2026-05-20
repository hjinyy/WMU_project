# Data Notes

This repository keeps only lightweight reference data in git.

Tracked here:
- `ieee30_edges.csv`

Not tracked here:
- raw waveform event Excel files (`*.xlsx`)
- MATLAB artifacts (`*.mat`, `*.slx`)
- large intermediate or local output directories

Raw waveform dataset used by the waveform-event pipeline:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw`
- WSL/Linux: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_raw`

Generated output directory used by the pipeline:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_data`
- WSL/Linux: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_data`

See `docs/waveform_dataset_notes.md` for dataset naming rules and schema expectations.
