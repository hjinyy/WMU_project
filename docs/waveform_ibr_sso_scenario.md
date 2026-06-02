# IBR-like SSO Scenario on the SPS IEEE 30-bus Model

## Scope

This document records the first automation pass for adding an IBR-like subsynchronous oscillation (SSO) scenario to the existing IEEE 30-bus Specialized Power Systems (SPS) model.

Important constraints:
- MATLAB version: **R2025b**
- The model is **SPS-based**, not Simscape physical-port based.
- **Simscape physical-port blocks must not be added** to the existing model.
- Existing SLG / ThreePhase / LoadSwitch events must be OFF before any SSO trial.

## What was implemented

The following MATLAB automation helpers were prepared:
- `scripts/matlab/wmu_inspect_ibr_sso_injection_options.m`
- `scripts/matlab/wmu_reset_all_events_for_ibr_sso.m`
- `scripts/matlab/wmu_apply_ibr_like_sso.m`
- `scripts/matlab/run_wmu_ibr_sso_smoke_test.m`

These scripts:
1. inspect SPS-compatible injection options,
2. verify every existing SLG/ThreePhase/LoadSwitch event is OFF,
3. block execution if no validated physical injection path is available.

## Inspection result

The inspection pass found:
- `powergui` is present and SPS libraries resolve correctly,
- many SPS load blocks are available and configurable,
- a `3-phase Programmable Source` exists in `powerlib_extras`, but it is a **Simulink control-signal block**, not an electrical injection block,
- no validated SPS-compatible controlled electrical source path was confirmed that can be inserted and driven automatically without moving outside the current model architecture.

The currently selected candidate in the raw inspection report was:
- `external_pq_dynamic_load`

However, this remains only a **candidate**, not a validated physical injection path, because time-varying P/Q modulation into the existing SPS network was not proven to work safely in the current model without further block-level redesign.

## Current status

A guarded smoke test was executed and intentionally stopped with:
- `physical injection path not available`

This is the correct fail-safe behavior. No synthetic waveform was created and no extended raw dataset was generated.

## Next required model work

Before the IBR-like SSO dataset can be built, one of the following must be implemented and validated inside the SPS model:
1. an SPS-compatible controlled electrical source path that accepts the programmable-source signal,
2. a proven SPS dynamic P/Q modulation path on a bus-connected load/source block,
3. or a small SPS-compatible injection subsystem designed specifically for 20–30 Hz oscillatory current/power injection.

Until one of those is available, the extended IBR-like SSO raw dataset must remain blocked.
