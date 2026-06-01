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
