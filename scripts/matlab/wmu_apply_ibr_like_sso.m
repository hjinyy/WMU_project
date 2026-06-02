function info = wmu_apply_ibr_like_sso(modelName, injectionBus, f_sso, amp_pu_or_amp, t_start, t_end, injectionMode)
if nargin < 7 || isempty(injectionMode)
    injectionMode = 'unavailable';
end
info = struct('modelName', modelName, 'InjectionBus', injectionBus, 'f_sso', f_sso, 'Amplitude', amp_pu_or_amp, 'StartTime', t_start, 'EndTime', t_end, 'InjectionMode', injectionMode, 'SourceBlockPath', '', 'Status', 'FAILED', 'Message', 'physical injection path not available');
error('physical injection path not available: only static SPS load parameters and a non-electrical programmable control block were found; no validated time-varying SPS electrical injection path is available.');
end
