function wmu_inspect_ibr_sso_injection_options()
modelPath = 'C:\Users\user\Documents\MATLAB\WMU_test\Thirtybussys.slx';
outDir = 'C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw_ibr_sso';
csvPath = fullfile(outDir, 'ibr_sso_injection_options_report.csv');
txtPath = fullfile(outDir, 'ibr_sso_injection_options_report.txt');
[~, modelName] = fileparts(modelPath);
load_system(modelPath);
cleanupObj = onCleanup(@() bdclose(modelName)); %#ok<NASGU>
rows = struct([]);
rows = [rows; local_scan_named_blocks(modelName, 'powergui', 'priority_check')]; %#ok<AGROW>
rows = [rows; local_find_candidates(modelName, 'current_source', {'Current Source','Controlled Current Source','AC Current Source','Three-Phase Current Source','Three-Phase Programmable Voltage Source'})]; %#ok<AGROW>
rows = [rows; local_find_candidates(modelName, 'generator_source', {'Synchronous Machine','Three-Phase Source','Programmable Voltage Source','Controlled Voltage Source'})]; %#ok<AGROW>
rows = [rows; local_find_candidates(modelName, 'load_pq', {'Three-Phase Series RLC Load','Three-Phase Parallel RLC Load','Dynamic Load'})]; %#ok<AGROW>
rows = [rows; local_find_candidates(modelName, 'measurement_or_control', {'From Workspace','Sine Wave','Controlled Voltage Source','Controlled Current Source'})]; %#ok<AGROW>
T = struct2table(rows);
writetable(T, csvPath);
[fMode, fPath, reason, unused] = local_choose_mode(T);
fid = fopen(txtPath, 'w');
assert(fid >= 0, 'Cannot open txt report.');
cleanupTxt = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'MATLAB version: %s\n', version);
fprintf(fid, 'MATLAB root   : %s\n', matlabroot);
fprintf(fid, 'Simscape physical port blocks used: No\n');
fprintf(fid, 'Selected injectionMode: %s\n', fMode);
fprintf(fid, 'Selected block path   : %s\n', fPath);
fprintf(fid, 'Selection reason      : %s\n', reason);
fprintf(fid, 'Unused / rejected paths and reasons:\n');
for i = 1:height(unused)
    fprintf(fid, '  - %s | %s | %s\n', unused.BlockPath{i}, unused.MaskType{i}, unused.RejectionReason{i});
end
fprintf(fid, '\nFull candidate table written to %s\n', csvPath);
end

function rows = local_scan_named_blocks(modelName, name, group)
rows = struct([]);
hits = find_system(modelName, 'LookUnderMasks','all', 'FollowLinks','on', 'Name', name);
for i = 1:numel(hits)
    rows = [rows; local_row(hits{i}, group)]; %#ok<AGROW>
end
end

function rows = local_find_candidates(modelName, group, maskNames)
rows = struct([]);
allBlocks = find_system(modelName, 'LookUnderMasks','all', 'FollowLinks','on', 'Type', 'Block');
for i = 1:numel(allBlocks)
    blk = allBlocks{i};
    maskType = local_safe_get(blk, 'MaskType');
    blockType = local_safe_get(blk, 'BlockType');
    if any(strcmpi(maskType, maskNames)) || any(strcmpi(blockType, maskNames)) || any(contains(lower(maskType), lower(maskNames)))
        rows = [rows; local_row(blk, group)]; %#ok<AGROW>
    end
end
end

function r = local_row(blk, group)
params = local_get_dialog_names(blk);
r = struct();
r.Group = group;
r.BlockPath = blk;
r.BlockName = local_safe_get(blk, 'Name');
r.BlockType = local_safe_get(blk, 'BlockType');
r.MaskType = local_safe_get(blk, 'MaskType');
r.LinkStatus = local_safe_get(blk, 'LinkStatus');
r.HasPhysicalPorts = local_has_physical_ports(blk);
r.DialogParameters = strjoin(params, '|');
r.PortConnectivity = local_port_summary(blk);
r.RejectionReason = '';
end

function [mode, path, reason, unused] = local_choose_mode(T)
mode = 'unavailable'; path = ''; reason = 'physical injection path not available';
unused = T;
if any(strcmpi(T.Group, 'load_pq') & contains(T.DialogParameters, 'ActivePower') & contains(T.DialogParameters, 'InductivePower'))
    idx = find(strcmpi(T.Group, 'load_pq') & contains(T.DialogParameters, 'ActivePower') & contains(T.DialogParameters, 'InductivePower'), 1, 'first');
    mode = 'external_pq_dynamic_load'; path = T.BlockPath{idx}; reason = 'SPS-compatible load block with configurable ActivePower/InductivePower found; suitable for P/Q oscillation injection.';
end
if strcmp(mode, 'unavailable') && any(strcmpi(T.Group, 'generator_source') & contains(T.DialogParameters, 'External'))
    idx = find(strcmpi(T.Group, 'generator_source') & contains(T.DialogParameters, 'External'), 1, 'first');
    mode = 'existing_generator_perturbation'; path = T.BlockPath{idx}; reason = 'Existing SPS source/generator exposes an external or programmable control input.';
end
if strcmp(mode, 'unavailable') && any(strcmpi(T.Group, 'current_source') & ~T.HasPhysicalPorts)
    idx = find(strcmpi(T.Group, 'current_source') & ~T.HasPhysicalPorts, 1, 'first');
    mode = 'sps_compatible_current_source'; path = T.BlockPath{idx}; reason = 'SPS-compatible current/source-like block found without physical ports.';
end
unused.RejectionReason(:) = {'not selected'};
if ~isempty(path)
    unused.RejectionReason(strcmp(unused.BlockPath, path)) = {'selected'};
end
end

function tf = local_has_physical_ports(blk)
    tf = false;
    try
        ph = get_param(blk, 'PortHandles');
        f = fieldnames(ph);
        for i = 1:numel(f)
            v = ph.(f{i});
            if ~isempty(v) && contains(lower(f{i}), 'rconn')
                tf = true; return;
            end
        end
    catch
    end
end

function params = local_get_dialog_names(blk)
    try
        s = get_param(blk, 'DialogParameters');
        params = fieldnames(s);
    catch
        params = {};
    end
end

function txt = local_port_summary(blk)
    txt = '';
    try
        pc = get_param(blk, 'PortConnectivity');
        parts = cell(numel(pc),1);
        for i = 1:numel(pc)
            parts{i} = sprintf('Port%d:%s', i, pc(i).Type);
        end
        txt = strjoin(parts, ' | ');
    catch
        txt = '(unavailable)';
    end
end

function v = local_safe_get(blk, name)
    try
        v = get_param(blk, name);
        if isstring(v), v = char(v); end
        if isnumeric(v), v = mat2str(v); end
    catch
        v = '(unavailable)';
    end
end
