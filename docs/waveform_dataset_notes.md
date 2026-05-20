# Waveform Dataset Notes

## Raw-data location

The raw 3-phase waveform event files used for this workflow live outside the repository:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw`
- WSL/Linux: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_raw`

The generated output directory used in this run is:
- Windows: `C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_data`
- WSL/Linux: `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_data`

## File naming rules

Expected xlsx patterns:
- `SLG_Fault_Bus01.xlsx`
- `ThreePhase_Fault_Bus01.xlsx`
- `LoadSwitch_Bus02.xlsx`
- `LoadSwitch15pct_Bus02.xlsx`
- `Normal_Case01.xlsx`

Parsing rules:
- `EventType` is parsed from the filename prefix.
- `TargetBus` is parsed from `_BusXX`.
- `Normal` cases do not require a target bus and are stored with `TargetBus = NaN`.
- `LoadSwitch15pct_*` is normalized into `EventType = LoadSwitch` while preserving the original variant string in metadata.

## Dataset composition detected on 2026-05-20

- Total xlsx files: 93
- Normal: 3
- LoadSwitch: 30
- SLG_Fault: 30
- ThreePhase_Fault: 30

## Expected workbook schema

The current pipeline expects one sheet per file and a time-domain table containing:
- `Time`
- `Va_<bus>`, `Vb_<bus>`, `Vc_<bus>`
- `Ia_<bus>`, `Ib_<bus>`, `Ic_<bus>`

for each observed bus.

## Git policy

The repository does **not** track:
- raw `.xlsx` event files,
- MATLAB `.mat` or `.slx` files,
- large intermediate outputs,
- local Windows-path scratch outputs.

Only code, docs, and compact summary reports/figures are kept in git.
