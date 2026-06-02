# IBR-like SSO Analysis Status

This folder is reserved for compact git-trackable outputs related to the IBR-like SSO scenario.

Current status:
- SPS-compatible injection options were inspected under MATLAB R2025b.
- Existing fault and LoadSwitch events can be reset and verified OFF.
- A guarded smoke-test gate is in place.
- **No extended IBR-like SSO raw dataset has been generated yet**, because no validated physical injection path was confirmed in the current SPS model without violating the no-Simscape-physical-port constraint.

Raw inspection outputs remain outside git under:
- `WMU_batch_raw_ibr_sso/`

See `docs/waveform_ibr_sso_scenario.md` for details.
