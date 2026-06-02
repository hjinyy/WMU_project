function run_wmu_ibr_sso_smoke_test()
modelPath = 'C:\Users\user\Documents\MATLAB\WMU_test\Thirtybussys.slx';
outDir = 'C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw_ibr_sso';
summaryPath = fullfile(outDir, 'smoke_tests', 'ibr_sso_smoke_test_summary.csv');
if ~exist(fullfile(outDir,'smoke_tests'),'dir'), mkdir(fullfile(outDir,'smoke_tests')); end
[~, modelName] = fileparts(modelPath);
load_system(modelPath); cleanupObj = onCleanup(@() bdclose(modelName)); %#ok<NASGU>
eventReport = fullfile(outDir, 'event_off_check_report.csv');
ok = wmu_reset_all_events_for_ibr_sso(modelName, eventReport);
row = struct('CaseName','IBR_SSO_Bus05_f25_smoke','InjectionBus',5,'f_sso',25,'InjectionMode','unavailable','EventsOffOK',ok,'Status','FAILED','Message','physical injection path not available','WaveformFigure','', 'FFT_Figure','');
writetable(struct2table(row, 'AsArray', true), summaryPath);
error('IBR-like SSO smoke test aborted: physical injection path not available in current SPS model/library setup.');
end
