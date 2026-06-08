# Waveform Dataset Notes

## Raw-data location

The raw 3-phase waveform event files used for this workflow live outside the repository:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw`
- WSL/Linux: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_raw`

The generated output directory used in this run is:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_data`
- WSL/Linux: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_data`

## File naming rules

Expected event-file patterns:
- `SLG_Fault_Bus01.xlsx`
- `ThreePhase_Fault_Bus01.xlsx`
- `LoadSwitch_Bus02.csv`
- `Normal_Case01.xlsx`

Parsing rules:
- `EventType` is parsed from the filename prefix.
- `TargetBus` is parsed from `_BusXX`.
- `Normal` cases do not require a target bus and are stored with `TargetBus = NaN`.
- The current pipeline accepts both `.xlsx` and `.csv` event files in the same raw directory.

## Dataset composition detected on 2026-05-20

- Total raw event files: 83
- Normal: 3
- LoadSwitch: 20
- SLG_Fault: 30
- ThreePhase_Fault: 30

## LoadSwitch regeneration rule

The original 30 LoadSwitch cases were discarded because they used an unsuitable fixed condition.
The current LoadSwitch cases were regenerated as **15% abrupt load increases** based on the existing IEEE 30-bus loads:
- `P_add = 0.15 * Pd / 100`
- `QL_add = 0.15 * Qd / 100`
- `Qc = 0`
- `baseMVA = 100`

Target buses are only the buses with existing Pd or Qd:
`2, 3, 4, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 29, 30`

Excluded buses without load:
`1, 5, 6, 9, 11, 13, 22, 25, 27, 28`

## Expected wide-format schema

Each raw event file is expected to contain:
- `Time`
- `Va_<bus>`, `Vb_<bus>`, `Vc_<bus>`
- `Ia_<bus>`, `Ib_<bus>`, `Ic_<bus>`

for each observed bus.

## Git policy

The repository does **not** track:
- raw `.xlsx` or `.csv` event files,
- MATLAB `.mat` or `.slx` files,
- large intermediate outputs,
- local Windows-path scratch outputs.

Only code, docs, and compact summary reports/figures are kept in git.

## SLG dataset regeneration note (2026-06-01)

- The original tracked SLG raw files were archived under `archive_bad_slg_*` after they were found to be effectively indistinguishable from `Normal`.
- A direct exact-parameter MATLAB R2025b path regenerated `SLG_Fault_Bus01.csv` ~ `SLG_Fault_Bus30.csv`.
- The rebuilt raw dataset now returns to `83` event files:
  - `Normal = 3`
  - `LoadSwitch = 20`
  - `ThreePhase_Fault = 30`
  - `SLG_Fault = 30`
- The regenerated SLG data no longer collides with the Normal feature hash, but the current Python classifier still swaps `SLG_Fault` and `ThreePhase_Fault` almost completely.
- Therefore the current dataset should be considered repaired enough for continued diagnostics, but not yet sufficient for a final SLG-vs-ThreePhase classification claim.

## IBR-background dataset generation note (2026-06-03)

A new modified IEEE 30-bus dataset was generated from the working-copy model:

- Source model: `C:\Users\user\Documents\MATLAB\WMU_final\Thirtybussys_WMU_IBR.slx`
- Working copy: `C:\Users\user\Documents\MATLAB\WMU_final\Thirtybussys_WMU_IBR_batch.slx`
- Raw output: `C:\Users\user\Documents\MATLAB\WMU_final\WMU_batch_raw_ibr_background`
- Analysis output: `C:\Users\user\Documents\MATLAB\WMU_final\WMU_batch_data_ibr_background`

The source model was copied before execution and was not saved. The working model was verified under MATLAB R2024a and configured with `StopTime = 0.5`.

New event-file patterns:

- `SSO_Normal_Case01.csv` ... `SSO_Normal_Case03.csv`
- `SSO_LoadSwitch_Bus02.csv` ... `SSO_LoadSwitch_Bus30.csv` for the 21 configured load buses
- `SSO_SLG_Fault_Bus01.csv` ... `SSO_SLG_Fault_Bus30.csv`
- `SSO_ThreePhase_Fault_Bus01.csv` ... `SSO_ThreePhase_Fault_Bus30.csv`

Dataset composition:

- `SSO_Normal = 3`
- `SSO_LoadSwitch = 21`
- `SSO_SLG_Fault = 30`
- `SSO_ThreePhase_Fault = 30`
- Total CSV files: `84`

All cases include the IBR-like SSO background. LoadSwitch cases additionally enable exactly one `LoadSwitch<bus>` breaker at `0.1 s`; SLG/three-phase fault cases additionally enable exactly one `SLG<bus>` fault from `0.3 s` to `0.36 s`.

Before LoadSwitch cases, `LoadAdd<bus>` blocks were set from existing `Load<bus>` values:

- `ActivePower_LoadAdd = 0.15 * ActivePower_Load`
- `InductivePower_LoadAdd = 0.15 * InductivePower_Load`
- `CapacitivePower_LoadAdd = 0`

Integrity reports are stored outside git in the raw output directory:

- `model_integrity_report.csv/.txt`
- `loadadd_15pct_setting_report.csv`
- `sanity_check_summary.csv`
- `dataset_metadata.csv`
- `dataset_integrity_report.csv/.txt`

An additional diagnostics pass is tracked under
`results/waveform_ibr_background_diagnostics/`. It reads the existing feature
tables only; the 84 raw CSV files are neither modified nor regenerated. The
result should be interpreted with the dataset's narrow design in mind: three
Normal cases, one SSO condition, one 15% load-switch magnitude, fixed fault
parameters, no noise, and deterministic simulation conditions.
