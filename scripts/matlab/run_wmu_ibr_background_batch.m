function run_wmu_ibr_background_batch(mode)
%RUN_WMU_IBR_BACKGROUND_BATCH Build IBR-background WMU waveform dataset.
%
% MATLAB R2024a entry point. Creates a working copy of the modified IEEE
% 30-bus Simulink model, verifies the expected blocks/parameters, sets
% LoadAdd blocks to 15% of existing load P/QL, runs a 4-case sanity set,
% then runs the 84-case batch only if sanity passes. The original model is
% never saved.
%
% Usage:
%   run_wmu_ibr_background_batch          % full workflow
%   run_wmu_ibr_background_batch('sanity')% preflight + 4 sanity cases only
%   run_wmu_ibr_background_batch('full')  % same as default

if nargin < 1 || isempty(mode)
    mode = 'full';
end
mode = char(mode);

cfg = local_config();
if ~exist(cfg.rawDir, 'dir'), mkdir(cfg.rawDir); end
fprintf('MATLAB version: %s\n', version);
fprintf('MATLAB root   : %s\n', matlabroot);
fprintf('Source model  : %s\n', cfg.sourceModel);
fprintf('Working model : %s\n', cfg.workModel);
fprintf('Raw output    : %s\n', cfg.rawDir);

if exist(cfg.workModel, 'file')
    delete(cfg.workModel);
end
copyfile(cfg.sourceModel, cfg.workModel, 'f');

[~, modelName] = fileparts(cfg.workModel);
load_system(cfg.workModel);
cleanupObj = onCleanup(@() local_cleanup_model(modelName)); %#ok<NASGU>
set_param(modelName, 'StopTime', '0.5');
local_set_powergui_frequency(modelName, 50);

integrity = local_verify_model(modelName, cfg);
writetable(integrity.table, cfg.modelIntegrityCsv);
local_write_text(cfg.modelIntegrityTxt, integrity.text);
fprintf('%s\n', integrity.text);
if ~integrity.ok
    error('Model integrity failed. Batch is intentionally blocked. See %s', cfg.modelIntegrityTxt);
end

loadaddReport = local_configure_loadadd_15pct(modelName, cfg);
writetable(loadaddReport, cfg.loadaddReportCsv);
if any(~strcmp(loadaddReport.Status, 'OK'))
    error('LoadAdd 15%% configuration failed. See %s', cfg.loadaddReportCsv);
end

save_system(modelName, cfg.workModel);

sanity = local_run_cases(modelName, cfg, local_build_sanity_cases(cfg), true);
writetable(sanity.summary, cfg.sanitySummaryCsv);
if any(~strcmp(sanity.summary.Status, 'OK')) || any(~strcmp(sanity.summary.SanityStatus, 'PASS'))
    error('Sanity check failed. Full 84-case batch is blocked. See %s', cfg.sanitySummaryCsv);
end

if strcmpi(mode, 'sanity')
    fprintf('Sanity-only mode complete. Full batch not requested.\n');
    return;
end

batch = local_run_cases(modelName, cfg, local_build_all_cases(cfg), false);
writetable(batch.metadata, cfg.datasetMetadataCsv);

integrityDataset = local_check_dataset_integrity(cfg);
writetable(integrityDataset.table, cfg.datasetIntegrityCsv);
local_write_text(cfg.datasetIntegrityTxt, integrityDataset.text);
fprintf('%s\n', integrityDataset.text);
if ~integrityDataset.ok
    warning('Dataset integrity has failures. See %s', cfg.datasetIntegrityTxt);
end
end

function cfg = local_config()
cfg = struct();
cfg.sourceModel = 'C:\Users\user\Documents\MATLAB\WMU_final\Thirtybussys_WMU_IBR.slx';
cfg.workModel = 'C:\Users\user\Documents\MATLAB\WMU_final\Thirtybussys_WMU_IBR_batch.slx';
cfg.rawDir = 'C:\Users\user\Documents\MATLAB\WMU_final\WMU_batch_raw_ibr_background';
cfg.analysisDir = 'C:\Users\user\Documents\MATLAB\WMU_final\WMU_batch_data_ibr_background';
cfg.numBuses = 30;
cfg.loadBuses = [2 3 4 5 7 8 10 12 14 15 16 17 18 19 20 21 23 24 26 29 30];
cfg.sso = struct('Frequency',25,'StartTime',0.02,'EndTime',0.48,'P0',0.1,'Q0',0.05,'dP',0.05,'dQ',0.025);
cfg.loadSwitchTime = 0.1;
cfg.loadSwitchPct = 15;
cfg.faultStart = 0.3;
cfg.faultClear = 0.36;
cfg.modelIntegrityCsv = fullfile(cfg.rawDir, 'model_integrity_report.csv');
cfg.modelIntegrityTxt = fullfile(cfg.rawDir, 'model_integrity_report.txt');
cfg.loadaddReportCsv = fullfile(cfg.rawDir, 'loadadd_15pct_setting_report.csv');
cfg.sanitySummaryCsv = fullfile(cfg.rawDir, 'sanity_check_summary.csv');
cfg.datasetMetadataCsv = fullfile(cfg.rawDir, 'dataset_metadata.csv');
cfg.datasetIntegrityCsv = fullfile(cfg.rawDir, 'dataset_integrity_report.csv');
cfg.datasetIntegrityTxt = fullfile(cfg.rawDir, 'dataset_integrity_report.txt');
end

function local_cleanup_model(modelName)
try
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
catch
end
end

function info = local_verify_model(modelName, cfg)
rows = struct('Check','', 'Expected','', 'Observed','', 'Status','', 'Message','');
rows(1) = [];
add = @(check, expected, observed, status, message) local_add_row(check, expected, observed, status, message);

for bus = 1:cfg.numBuses
    name = sprintf('SLG%d', bus);
    [path, msg] = local_find_unique(modelName, name);
    if isempty(path)
        rows(end+1) = add(sprintf('SLG%d exists', bus), name, '', 'FAILED', msg); %#ok<AGROW>
        continue;
    end
    rows(end+1) = add(sprintf('SLG%d exists', bus), name, path, 'OK', ''); %#ok<AGROW>
    sourceType = local_safe_get(path, 'SourceType');
    maskType = local_safe_get(path, 'MaskType');
    rows(end+1) = add(sprintf('SLG%d SourceType', bus), 'Three-Phase Fault family', [sourceType ' | ' maskType], local_status(contains(lower([sourceType ' ' maskType]), 'fault')), ''); %#ok<AGROW>
    rows = local_expect_param(rows, path, sprintf('SLG%d FaultA', bus), 'FaultA', 'off');
    rows = local_expect_param(rows, path, sprintf('SLG%d FaultB', bus), 'FaultB', 'off');
    rows = local_expect_param(rows, path, sprintf('SLG%d FaultC', bus), 'FaultC', 'off');
    rows = local_expect_param(rows, path, sprintf('SLG%d GroundFault', bus), 'GroundFault', 'off');
    rows = local_expect_param(rows, path, sprintf('SLG%d SwitchTimes', bus), 'SwitchTimes', '[0.3 0.36]');
end

for bus = cfg.loadBuses
    name = sprintf('LoadSwitch%d', bus);
    [path, msg] = local_find_unique(modelName, name);
    if isempty(path)
        rows(end+1) = add(sprintf('LoadSwitch%d exists', bus), name, '', 'FAILED', msg); %#ok<AGROW>
        continue;
    end
    rows(end+1) = add(sprintf('LoadSwitch%d exists', bus), name, path, 'OK', ''); %#ok<AGROW>
    rows = local_expect_param(rows, path, sprintf('LoadSwitch%d InitialState', bus), 'InitialState', 'open');
    rows = local_expect_param(rows, path, sprintf('LoadSwitch%d SwitchA', bus), 'SwitchA', 'off');
    rows = local_expect_param(rows, path, sprintf('LoadSwitch%d SwitchB', bus), 'SwitchB', 'off');
    rows = local_expect_param(rows, path, sprintf('LoadSwitch%d SwitchC', bus), 'SwitchC', 'off');
    rows = local_expect_param(rows, path, sprintf('LoadSwitch%d SwitchTimes', bus), 'SwitchTimes', '[0.1]');
end

for bus = cfg.loadBuses
    for prefix = {sprintf('Load%d', bus), sprintf('LoadAdd%d', bus)}
        [path, msg] = local_find_unique(modelName, prefix{1});
        rows(end+1) = add(sprintf('%s exists', prefix{1}), prefix{1}, path, local_status(~isempty(path)), msg); %#ok<AGROW>
    end
end

for bus = 1:cfg.numBuses
    for prefix = {'V','I'}
        varName = sprintf('%s_%d', prefix{1}, bus);
        hits = find_system(modelName, 'LookUnderMasks','all','FollowLinks','on','RegExp','off','BlockType','ToWorkspace','VariableName',varName);
        status = local_status(numel(hits) == 1);
        rows(end+1) = add(sprintf('ToWorkspace %s', varName), '1 block', sprintf('%d', numel(hits)), status, local_join_hits(hits)); %#ok<AGROW>
        if numel(hits) == 1
            fmt = local_safe_get(hits{1}, 'SaveFormat');
            rows(end+1) = add(sprintf('ToWorkspace %s SaveFormat', varName), 'Timeseries accepted', fmt, local_status(strcmpi(fmt,'Timeseries')), 'export reads timeseries Time/Data'); %#ok<AGROW>
        end
    end
end

dynLoads = local_find_dynamic_loads(modelName);
rows(end+1) = add('IBR Dynamic Load count', '5 Three-Phase Dynamic Load blocks', sprintf('%d', numel(dynLoads)), local_status(numel(dynLoads)==5), local_join_hits(dynLoads)); %#ok<AGROW>
for i = 1:numel(dynLoads)
    p = dynLoads{i};
    rows = local_expect_any_param(rows, p, sprintf('IBR Dynamic Load %d external PQ', i), {'ExternalControl','ExternalPQ','ExternalControlPQ','ExternalControlOfPQ','PQExternalControl'}, {'on','1','true'});
    rows = local_expect_param(rows, p, sprintf('IBR Dynamic Load %d NominalVoltage', i), 'NominalVoltage', '[1 50]');
    rows = local_expect_param(rows, p, sprintf('IBR Dynamic Load %d ActiveReactivePowers', i), 'ActiveReactivePowers', '[0 0]');
    rows = local_expect_param(rows, p, sprintf('IBR Dynamic Load %d PositiveSequence', i), 'PositiveSequence', '[1 0]');
    rows = local_expect_param(rows, p, sprintf('IBR Dynamic Load %d Tfilter', i), 'Tfilter', '1e-4');
end
rows(end+1) = add('MATLAB Function PQ oscillation', '25Hz t1=0.02 t2=0.48 P0=0.1 Q0=0.05 dP=0.05 dQ=0.025', local_check_matlab_functions(modelName, cfg), local_status(contains(local_check_matlab_functions(modelName, cfg), 'PASS')), ''); %#ok<AGROW>
rows(end+1) = add('StopTime', '0.5', char(get_param(modelName,'StopTime')), local_status(local_num_equal(get_param(modelName,'StopTime'), 0.5)), ''); %#ok<AGROW>

T = struct2table(rows);
failed = strcmp(T.Status, 'FAILED');
info = struct();
info.table = T;
info.ok = ~any(failed);
info.text = sprintf('Model integrity: %s | OK=%d FAILED=%d\n', ternary(info.ok,'PASS','FAIL'), nnz(strcmp(T.Status,'OK')), nnz(failed));
end

function rows = local_expect_param(rows, blockPath, checkName, paramName, expected)
observed = local_safe_get(blockPath, paramName);
if isempty(observed)
    rows(end+1) = local_add_row(checkName, expected, '', 'FAILED', sprintf('Parameter %s not found', paramName));
else
    rows(end+1) = local_add_row(checkName, expected, observed, local_status(local_value_equal(observed, expected)), '');
end
end

function rows = local_expect_any_param(rows, blockPath, checkName, candidates, expectedVals)
params = fieldnames(local_param_struct(blockPath, 'DialogParameters'));
chosen = local_find_param(params, candidates);
if isempty(chosen)
    rows(end+1) = local_add_row(checkName, strjoin(expectedVals, '|'), '', 'FAILED', 'No candidate parameter present');
else
    observed = local_safe_get(blockPath, chosen);
    ok = any(strcmpi(strtrim(observed), expectedVals));
    rows(end+1) = local_add_row(checkName, strjoin(expectedVals, '|'), sprintf('%s=%s', chosen, observed), local_status(ok), '');
end
end

function T = local_configure_loadadd_15pct(modelName, cfg)
rows = struct('Bus',{},'ExistingLoadBlock',{},'LoadAddBlock',{},'ExistingActivePower',{},'ExistingInductivePower',{},'ComputedActivePower15pct',{},'ComputedInductivePower15pct',{},'AppliedActivePower',{},'AppliedInductivePower',{},'AppliedCapacitivePower',{},'Status',{},'Message',{});
for bus = cfg.loadBuses
    status = 'OK'; msg = '';
    existingP = NaN; existingQL = NaN; compP = NaN; compQL = NaN; appP = NaN; appQL = NaN; appQc = NaN;
    [loadPath, msg1] = local_find_unique(modelName, sprintf('Load%d', bus));
    [addPath, msg2] = local_find_unique(modelName, sprintf('LoadAdd%d', bus));
    try
        if isempty(loadPath) || isempty(addPath), error('%s %s', msg1, msg2); end
        [pName, qlName, qcName] = local_load_param_names(loadPath);
        [apName, aqlName, aqcName] = local_load_param_names(addPath);
        existingP = str2double(local_safe_get(loadPath, pName));
        existingQL = str2double(local_safe_get(loadPath, qlName));
        if isnan(existingP) || isnan(existingQL), error('Could not parse existing P/QL for bus %d', bus); end
        compP = 0.15 * existingP;
        compQL = 0.15 * existingQL;
        set_param(addPath, apName, num2str(compP, '%.12g'));
        set_param(addPath, aqlName, num2str(compQL, '%.12g'));
        set_param(addPath, aqcName, '0');
        local_set_if_param(addPath, {'NominalVoltage','Vn','Vnom'}, '1');
        local_set_if_param(addPath, {'NominalFrequency','fn','Frequency','Freq'}, '50');
        appP = str2double(local_safe_get(addPath, apName));
        appQL = str2double(local_safe_get(addPath, aqlName));
        appQc = str2double(local_safe_get(addPath, aqcName));
        if max(abs([appP-compP, appQL-compQL, appQc-0])) > 1e-9
            error('Applied values mismatch for LoadAdd%d', bus);
        end
        msg = 'Applied from existing Load P/QL';
    catch ME
        status = 'FAILED'; msg = local_error(ME);
    end
    rows(end+1) = struct('Bus',bus,'ExistingLoadBlock',loadPath,'LoadAddBlock',addPath,'ExistingActivePower',existingP,'ExistingInductivePower',existingQL,'ComputedActivePower15pct',compP,'ComputedInductivePower15pct',compQL,'AppliedActivePower',appP,'AppliedInductivePower',appQL,'AppliedCapacitivePower',appQc,'Status',status,'Message',msg); %#ok<AGROW>
end
T = struct2table(rows);
end

function result = local_run_cases(modelName, cfg, cases, sanityMode)
metaRows = struct('CaseName',{},'EventType',{},'TargetBus',{},'IBR_SSO_Background',{},'SSO_Frequency',{},'SSO_StartTime',{},'SSO_EndTime',{},'LoadSwitchTime',{},'LoadSwitchPct',{},'LoadAddActivePower',{},'LoadAddInductivePower',{},'FaultStartTime',{},'FaultClearTime',{},'FaultType',{},'OutputFile',{},'Status',{},'Message',{});
sanityRows = struct('CaseName',{},'EventType',{},'TargetBus',{},'OutputFile',{},'Status',{},'Message',{},'SanityStatus',{},'SanityMessage',{});
for idx = 1:numel(cases)
    c = cases(idx);
    fprintf('[%03d/%03d] %s\n', idx, numel(cases), c.CaseName);
    status = 'OK'; message = 'Simulation and export completed.'; sanityStatus = 'PASS'; sanityMessage = '';
    loadP = NaN; loadQL = NaN; faultType = '';
    try
        evalin('base', 'clear V_* I_*');
        local_reset_all_events(modelName, cfg);
        if strcmp(c.EventType, 'SSO_LoadSwitch')
            [loadP, loadQL] = local_enable_loadswitch_case(modelName, c.TargetBus, cfg);
        elseif strcmp(c.EventType, 'SSO_SLG_Fault')
            local_enable_fault_case(modelName, c.TargetBus, false, cfg); faultType = 'SLG';
        elseif strcmp(c.EventType, 'SSO_ThreePhase_Fault')
            local_enable_fault_case(modelName, c.TargetBus, true, cfg); faultType = 'ThreePhase';
        end
        simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');
        local_export_simout_to_csv(simOut, c.OutputFile, 1:cfg.numBuses);
        if sanityMode
            [sanityStatus, sanityMessage] = local_sanity_check_file(c, loadP, loadQL, cfg);
        end
    catch ME
        status = 'FAILED'; message = local_error(ME); sanityStatus = 'FAIL'; sanityMessage = message;
        fprintf(2, '  FAILED: %s\n', message);
    end
    metaRows(end+1) = struct('CaseName',c.CaseName,'EventType',c.EventType,'TargetBus',c.TargetBus,'IBR_SSO_Background',true,'SSO_Frequency',cfg.sso.Frequency,'SSO_StartTime',cfg.sso.StartTime,'SSO_EndTime',cfg.sso.EndTime,'LoadSwitchTime',local_ifelse(strcmp(c.EventType,'SSO_LoadSwitch'),cfg.loadSwitchTime,NaN),'LoadSwitchPct',local_ifelse(strcmp(c.EventType,'SSO_LoadSwitch'),cfg.loadSwitchPct,NaN),'LoadAddActivePower',loadP,'LoadAddInductivePower',loadQL,'FaultStartTime',local_ifelse(contains(c.EventType,'Fault'),cfg.faultStart,NaN),'FaultClearTime',local_ifelse(contains(c.EventType,'Fault'),cfg.faultClear,NaN),'FaultType',faultType,'OutputFile',c.OutputFile,'Status',status,'Message',message); %#ok<AGROW>
    if sanityMode
        sanityRows(end+1) = struct('CaseName',c.CaseName,'EventType',c.EventType,'TargetBus',c.TargetBus,'OutputFile',c.OutputFile,'Status',status,'Message',message,'SanityStatus',sanityStatus,'SanityMessage',sanityMessage); %#ok<AGROW>
    end
end
result = struct();
result.metadata = struct2table(metaRows);
if sanityMode, result.summary = struct2table(sanityRows); else, result.summary = table(); end
end

function local_reset_all_events(modelName, cfg)
for bus = 1:cfg.numBuses
    slg = local_find_required(modelName, sprintf('SLG%d', bus));
    set_param(slg, 'FaultA', 'off', 'FaultB', 'off', 'FaultC', 'off', 'GroundFault', 'off', 'SwitchTimes', '[0.3 0.36]');
end
for bus = cfg.loadBuses
    brk = local_find_required(modelName, sprintf('LoadSwitch%d', bus));
    set_param(brk, 'SwitchA', 'off', 'SwitchB', 'off', 'SwitchC', 'off', 'InitialState', 'open', 'SwitchTimes', '[0.1]');
end
end

function [loadP, loadQL] = local_enable_loadswitch_case(modelName, bus, cfg)
brk = local_find_required(modelName, sprintf('LoadSwitch%d', bus));
set_param(brk, 'SwitchA', 'on', 'SwitchB', 'on', 'SwitchC', 'on', 'SwitchTimes', '[0.1]', 'InitialState', 'open');
assert(strcmpi(get_param(brk,'SwitchA'),'on') && strcmpi(get_param(brk,'SwitchB'),'on') && strcmpi(get_param(brk,'SwitchC'),'on'));
addPath = local_find_required(modelName, sprintf('LoadAdd%d', bus));
[pName, qlName] = local_load_param_names(addPath);
loadP = str2double(local_safe_get(addPath, pName));
loadQL = str2double(local_safe_get(addPath, qlName));
if isnan(loadP) || isnan(loadQL) || loadP <= 0
    error('LoadAdd%d 15%% values are invalid before LoadSwitch event.', bus);
end
end

function local_enable_fault_case(modelName, bus, threePhase, cfg)
slg = local_find_required(modelName, sprintf('SLG%d', bus));
if threePhase
    set_param(slg, 'FaultA','on','FaultB','on','FaultC','on','GroundFault','on','SwitchTimes','[0.3 0.36]');
    ok = strcmpi(get_param(slg,'FaultA'),'on') && strcmpi(get_param(slg,'FaultB'),'on') && strcmpi(get_param(slg,'FaultC'),'on') && strcmpi(get_param(slg,'GroundFault'),'on');
else
    set_param(slg, 'FaultA','on','FaultB','off','FaultC','off','GroundFault','on','SwitchTimes','[0.3 0.36]');
    ok = strcmpi(get_param(slg,'FaultA'),'on') && strcmpi(get_param(slg,'FaultB'),'off') && strcmpi(get_param(slg,'FaultC'),'off') && strcmpi(get_param(slg,'GroundFault'),'on');
end
if ~ok, error('Fault block verification failed for SLG%d', bus); end
end

function cases = local_build_sanity_cases(cfg)
cases = [
    local_case('SSO_Normal_Case01','SSO_Normal',NaN,fullfile(cfg.rawDir,'SSO_Normal_Case01.csv'))
    local_case('SSO_LoadSwitch_Bus05','SSO_LoadSwitch',5,fullfile(cfg.rawDir,'SSO_LoadSwitch_Bus05.csv'))
    local_case('SSO_SLG_Fault_Bus05','SSO_SLG_Fault',5,fullfile(cfg.rawDir,'SSO_SLG_Fault_Bus05.csv'))
    local_case('SSO_ThreePhase_Fault_Bus05','SSO_ThreePhase_Fault',5,fullfile(cfg.rawDir,'SSO_ThreePhase_Fault_Bus05.csv'))
];
end

function cases = local_build_all_cases(cfg)
cases = struct('CaseName',{},'EventType',{},'TargetBus',{},'OutputFile',{});
for k = 1:3
    name = sprintf('SSO_Normal_Case%02d', k);
    cases(end+1) = local_case(name,'SSO_Normal',NaN,fullfile(cfg.rawDir,[name '.csv'])); %#ok<AGROW>
end
for bus = cfg.loadBuses
    name = sprintf('SSO_LoadSwitch_Bus%02d', bus);
    cases(end+1) = local_case(name,'SSO_LoadSwitch',bus,fullfile(cfg.rawDir,[name '.csv'])); %#ok<AGROW>
end
for bus = 1:cfg.numBuses
    name = sprintf('SSO_SLG_Fault_Bus%02d', bus);
    cases(end+1) = local_case(name,'SSO_SLG_Fault',bus,fullfile(cfg.rawDir,[name '.csv'])); %#ok<AGROW>
end
for bus = 1:cfg.numBuses
    name = sprintf('SSO_ThreePhase_Fault_Bus%02d', bus);
    cases(end+1) = local_case(name,'SSO_ThreePhase_Fault',bus,fullfile(cfg.rawDir,[name '.csv'])); %#ok<AGROW>
end
end

function c = local_case(name,event,bus,out)
c = struct('CaseName',name,'EventType',event,'TargetBus',bus,'OutputFile',out);
end

function local_export_simout_to_csv(simOut, outputFile, buses)
signals = struct('Name',{},'Kind',{},'Bus',{},'Time',{},'Data',{});
for bus = buses
    for kind = {'V','I'}
        name = sprintf('%s_%d', kind{1}, bus);
        raw = local_get_sim_var(simOut, name);
        [t, d] = local_extract_signal(raw, name);
        signals(end+1) = struct('Name',name,'Kind',kind{1},'Bus',bus,'Time',t,'Data',d); %#ok<AGROW>
    end
end
masterTime = signals(1).Time(:);
T = table(masterTime, 'VariableNames', {'Time'});
for i = 1:numel(signals)
    d = signals(i).Data;
    if size(d,2) < 3, error('%s does not have 3 phases', signals(i).Name); end
    a = local_align(signals(i).Time, d(:,1:3), masterTime, signals(i).Name);
    if strcmp(signals(i).Kind,'V'), names = {'Va','Vb','Vc'}; else, names = {'Ia','Ib','Ic'}; end
    for p = 1:3
        T.(sprintf('%s_%d', names{p}, signals(i).Bus)) = a(:,p);
    end
end
writetable(T, outputFile);
end

function value = local_get_sim_var(simOut, varName)
value = [];
try
    if isa(simOut, 'Simulink.SimulationOutput') && any(strcmp(simOut.who, varName))
        value = simOut.get(varName); return;
    end
catch
end
if evalin('base', sprintf('exist(''%s'',''var'')', varName)) == 1
    value = evalin('base', varName); return;
end
error('Missing To Workspace variable: %s', varName);
end

function [time, data] = local_extract_signal(raw, signalName)
if isa(raw, 'timeseries')
    time = raw.Time; data = squeeze(raw.Data);
elseif isstruct(raw) && isfield(raw,'time') && isfield(raw,'signals')
    time = raw.time; data = squeeze(raw.signals.values);
elseif isnumeric(raw)
    data = squeeze(raw); time = (0:size(data,1)-1).';
else
    error('Unsupported signal format for %s: %s', signalName, class(raw));
end
if ndims(data) > 2
    sz = size(data); data = reshape(data, sz(1), []);
end
if size(data,1) == 1 && size(data,2) > 1, data = data.'; end
if size(data,1) ~= numel(time) && size(data,2) == numel(time), data = data.'; end
if size(data,1) ~= numel(time), error('%s time/data shape mismatch', signalName); end
end

function a = local_align(t, d, mt, name)
t = t(:); mt = mt(:);
if numel(t) == numel(mt) && max(abs(double(t)-double(mt))) < 1e-9
    a = d; return;
end
a = interp1(double(t), double(d), double(mt), 'linear', 'extrap');
if any(~isfinite(a(:))), error('%s alignment produced nonfinite values', name); end
end

function [sanityStatus, sanityMessage] = local_sanity_check_file(c, loadP, loadQL, cfg)
try
    T = readtable(c.OutputFile, 'VariableNamingRule','preserve');
    if height(T) < 10, error('Too few samples'); end
    vals = T{:,2:end};
    if any(~isfinite(vals(:))), error('NaN/Inf in exported CSV'); end
    t = T.Time;
    if strcmp(c.EventType, 'SSO_Normal')
        ok = local_band_ratio_ok(t, T.Va_1, 20, 30);
        if ~ok, error('20-30 Hz background component not detected in Va_1'); end
        sanityMessage = '20-30 Hz IBR background component detected';
    elseif strcmp(c.EventType, 'SSO_LoadSwitch')
        if isnan(loadP) || isnan(loadQL) || loadP <= 0, error('LoadAdd5 15%% values not available'); end
        ok = local_change_after(T, cfg.loadSwitchTime, 5, false);
        if ~ok, error('LoadSwitch post-0.1s change not detected'); end
        sanityMessage = sprintf('LoadAdd5 P=%g QL=%g and post-0.1s change detected', loadP, loadQL);
    elseif strcmp(c.EventType, 'SSO_SLG_Fault')
        ok = local_change_after(T, cfg.faultStart, 5, true);
        if ~ok, error('SLG A-phase fault-window signature not detected'); end
        sanityMessage = 'A-phase 0.3-0.36s fault-window signature detected';
    else
        ok = local_change_after(T, cfg.faultStart, 5, false);
        if ~ok, error('Three-phase fault-window signature not detected'); end
        sanityMessage = '3-phase 0.3-0.36s fault-window signature detected';
    end
    sanityStatus = 'PASS';
catch ME
    sanityStatus = 'FAIL'; sanityMessage = local_error(ME);
end
end

function ok = local_band_ratio_ok(t, x, f1, f2)
t = double(t(:)); x = double(x(:));
dt = median(diff(t)); fs = 1/dt;
x = x - mean(x, 'omitnan');
if numel(x) < 32, ok = false; return; end
Y = abs(fft(x)).^2; f = (0:numel(x)-1)' * fs / numel(x);
mask = f >= f1 & f <= f2;
ok = sum(Y(mask)) > max(1e-12, 1e-7 * sum(Y(f > 0 & f <= fs/2)));
end

function ok = local_change_after(T, tEvent, bus, aOnly)
t = T.Time;
if aOnly
    cols = {sprintf('Va_%d',bus), sprintf('Ia_%d',bus)};
else
    cols = {sprintf('Va_%d',bus),sprintf('Vb_%d',bus),sprintf('Vc_%d',bus),sprintf('Ia_%d',bus),sprintf('Ib_%d',bus),sprintf('Ic_%d',bus)};
end
pre = t >= max(0, tEvent-0.08) & t < tEvent;
post = t >= tEvent & t <= min(t(end), tEvent+0.08);
preVals = T{pre, cols}; postVals = T{post, cols};
base = mean(abs(preVals(:)), 'omitnan');
delta = abs(mean(abs(postVals(:)), 'omitnan') - base);
ok = delta > max(1e-8, 1e-4 * max(base, 1));
end

function out = local_check_dataset_integrity(cfg)
files = dir(fullfile(cfg.rawDir, 'SSO_*.csv'));
rows = struct('FileName',{},'CaseName',{},'EventType',{},'ColumnCount',{},'RowCount',{},'HasTime',{},'HasAllSignals',{},'NaNCount',{},'InfCount',{},'Status',{},'Message',{});
expectedColumns = [{'Time'}, local_expected_signal_columns(cfg.numBuses)];
for i = 1:numel(files)
    fp = fullfile(files(i).folder, files(i).name);
    status = 'OK'; msg = '';
    colCount = NaN; rowCount = NaN; hasTime = false; hasAll = false; nanCount = NaN; infCount = NaN;
    try
        T = readtable(fp, 'VariableNamingRule','preserve');
        colCount = width(T); rowCount = height(T); hasTime = any(strcmp(T.Properties.VariableNames,'Time'));
        hasAll = all(ismember(expectedColumns, T.Properties.VariableNames));
        vals = T{:, :}; nanCount = nnz(isnan(vals(:))); infCount = nnz(isinf(vals(:)));
        if rowCount == 0 || colCount ~= numel(expectedColumns) || ~hasTime || ~hasAll || nanCount > 0 || infCount > 0
            status = 'FAILED'; msg = 'Column/time/signal/nonfinite integrity failure';
        end
    catch ME
        status = 'FAILED'; msg = local_error(ME);
    end
    [eventType, caseName] = local_event_from_name(files(i).name);
    rows(end+1) = struct('FileName',fp,'CaseName',caseName,'EventType',eventType,'ColumnCount',colCount,'RowCount',rowCount,'HasTime',hasTime,'HasAllSignals',hasAll,'NaNCount',nanCount,'InfCount',infCount,'Status',status,'Message',msg); %#ok<AGROW>
end
T = struct2table(rows);
counts = groupsummary(T, 'EventType');
expected = struct('SSO_Normal',3,'SSO_LoadSwitch',21,'SSO_SLG_Fault',30,'SSO_ThreePhase_Fault',30);
ok = height(T) == 84 && nnz(strcmp(T.Status,'FAILED')) == 0 && local_count_ok(T, expected);
text = sprintf('Dataset integrity: %s | CSV=%d | failed=%d\nCounts:\n%s\n', ternary(ok,'PASS','FAIL'), height(T), nnz(strcmp(T.Status,'FAILED')), evalc('disp(counts)'));
out = struct('table',T,'text',text,'ok',ok);
end

function cols = local_expected_signal_columns(n)
cols = {};
for bus = 1:n
    cols = [cols, {sprintf('Va_%d',bus),sprintf('Vb_%d',bus),sprintf('Vc_%d',bus),sprintf('Ia_%d',bus),sprintf('Ib_%d',bus),sprintf('Ic_%d',bus)}]; %#ok<AGROW>
end
end

function ok = local_count_ok(T, expected)
ok = true;
names = fieldnames(expected);
for i = 1:numel(names)
    ok = ok && nnz(strcmp(T.EventType, names{i})) == expected.(names{i});
end
end

function [eventType, caseName] = local_event_from_name(fileName)
[~, caseName] = fileparts(fileName);
if startsWith(caseName,'SSO_Normal'), eventType = 'SSO_Normal';
elseif startsWith(caseName,'SSO_LoadSwitch'), eventType = 'SSO_LoadSwitch';
elseif startsWith(caseName,'SSO_SLG_Fault'), eventType = 'SSO_SLG_Fault';
elseif startsWith(caseName,'SSO_ThreePhase_Fault'), eventType = 'SSO_ThreePhase_Fault';
else, eventType = 'UNKNOWN'; end
end

function dynLoads = local_find_dynamic_loads(modelName)
hits = find_system(modelName, 'LookUnderMasks','all','FollowLinks','on','RegExp','on','MaskType','.*Dynamic Load.*');
dynLoads = {};
for i = 1:numel(hits)
    mt = local_safe_get(hits{i}, 'MaskType');
    st = local_safe_get(hits{i}, 'SourceType');
    if contains(lower([mt ' ' st]), 'dynamic load')
        dynLoads{end+1} = hits{i}; %#ok<AGROW>
    end
end
dynLoads = unique(dynLoads);
end

function msg = local_check_matlab_functions(modelName, cfg)
try
    rt = sfroot;
    machine = rt.find('-isa','Stateflow.Machine','Name',modelName);
    charts = machine.find('-isa','Stateflow.EMChart');
    code = '';
    for i = 1:numel(charts), code = [code newline charts(i).Script]; end %#ok<AGROW>
    needles = {'25','0.02','0.48','0.1','0.05','0.025'};
    ok = all(cellfun(@(s) contains(code,s), needles)) && contains(lower(code),'sin');
    msg = ternary(ok, 'PASS: MATLAB Function contains expected 25Hz PQ constants', 'FAIL: expected PQ constants not found in MATLAB Function code');
catch ME
    msg = ['FAIL: ' local_error(ME)];
end
end

function local_set_powergui_frequency(modelName, f)
hits = find_system(modelName, 'LookUnderMasks','all','FollowLinks','on','RegExp','off','Name','powergui');
for i = 1:numel(hits)
    p = hits{i};
    params = fieldnames(local_param_struct(p, 'DialogParameters'));
    for j = 1:numel(params)
        nm = params{j}; low = lower(nm);
        if contains(low, 'frequency') || contains(low, 'fundamental')
            try, set_param(p, nm, num2str(f)); catch, end
        end
    end
end
end

function [pName, qlName, qcName] = local_load_param_names(blockPath)
params = unique([fieldnames(local_param_struct(blockPath,'DialogParameters')); fieldnames(local_param_struct(blockPath,'ObjectParameters'))]);
pName = local_find_param(params, {'ActivePower','P','ThreePhaseActivePower','Pn'});
qlName = local_find_param(params, {'InductivePower','QL','InductiveReactivePower','ReactivePower','Q'});
qcName = local_find_param(params, {'CapacitivePower','Qc','CapacitiveReactivePower'});
if isempty(pName) || isempty(qlName) || isempty(qcName)
    error('Load parameter names not found on %s (P=%s QL=%s Qc=%s)', blockPath, pName, qlName, qcName);
end
end

function local_set_if_param(blockPath, candidates, value)
params = unique([fieldnames(local_param_struct(blockPath,'DialogParameters')); fieldnames(local_param_struct(blockPath,'ObjectParameters'))]);
name = local_find_param(params, candidates);
if ~isempty(name), try, set_param(blockPath, name, value); catch, end, end
end

function name = local_find_param(params, candidates)
name = '';
normParams = cellfun(@local_norm, params, 'UniformOutput', false);
for i = 1:numel(candidates)
    idx = find(strcmp(normParams, local_norm(candidates{i})), 1);
    if ~isempty(idx), name = params{idx}; return; end
end
for i = 1:numel(candidates)
    needle = local_norm(candidates{i});
    idx = find(contains(normParams, needle), 1);
    if ~isempty(idx), name = params{idx}; return; end
end
end

function s = local_norm(s)
s = lower(regexprep(char(s), '[^a-z0-9]', ''));
end

function s = local_safe_get(blockPath, paramName)
try
    s = get_param(blockPath, paramName);
    if isnumeric(s), s = mat2str(s); end
    if isstring(s), s = char(s); end
catch
    s = '';
end
end

function params = local_param_struct(blockPath, kind)
try
    params = get_param(blockPath, kind);
    if ~isstruct(params), params = struct(); end
catch
    params = struct();
end
end

function [path, msg] = local_find_unique(modelName, blockName)
hits = find_system(modelName, 'LookUnderMasks','all','FollowLinks','on','RegExp','off','Name',blockName);
if numel(hits) == 1, path = hits{1}; msg = '';
elseif isempty(hits), path = ''; msg = 'missing';
else, path = ''; msg = sprintf('not unique (%d hits)', numel(hits)); end
end

function path = local_find_required(modelName, blockName)
[path, msg] = local_find_unique(modelName, blockName);
if isempty(path), error('Required block %s: %s', blockName, msg); end
end

function row = local_add_row(check, expected, observed, status, message)
row = struct('Check',char(check),'Expected',char(expected),'Observed',char(observed),'Status',char(status),'Message',char(message));
end

function status = local_status(ok)
if ok, status = 'OK'; else, status = 'FAILED'; end
end

function ok = local_value_equal(a, b)
sa = lower(strtrim(char(a))); sb = lower(strtrim(char(b)));
ok = strcmp(sa, sb) || (local_num_equal(sa, str2double(sb)) && ~isnan(str2double(sb)));
end

function ok = local_num_equal(a, b)
x = str2double(char(a)); ok = ~isnan(x) && abs(x-b) < 1e-9;
end

function out = local_join_hits(hits)
if isempty(hits), out = ''; else, out = strjoin(cellstr(hits), ' | '); end
end

function local_write_text(path, txt)
fid = fopen(path, 'w'); fprintf(fid, '%s', txt); fclose(fid);
end

function s = local_error(ME)
s = regexprep(sprintf('%s | %s', ME.identifier, ME.message), '[\r\n]+', ' ');
end

function y = ternary(cond, a, b)
if cond, y = a; else, y = b; end
end

function y = local_ifelse(cond, a, b)
if cond, y = a; else, y = b; end
end
