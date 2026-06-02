function ok = wmu_reset_all_events_for_ibr_sso(modelName, outCsv)
if nargin < 2 || isempty(outCsv)
    outCsv = 'C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw_ibr_sso\event_off_check_report.csv';
end
rows = struct([]);
ok = true;
for bus = 1:30
    slg = sprintf('%s/SLG%d', modelName, bus);
    set_param(slg, 'FaultA', 'off');
    set_param(slg, 'FaultB', 'off');
    set_param(slg, 'FaultC', 'off');
    set_param(slg, 'GroundFault', 'off');
    r = struct('BlockType','SLG','BlockPath',slg,'AorSwitchA',char(get_param(slg,'FaultA')),'BorSwitchB',char(get_param(slg,'FaultB')),'CorSwitchC',char(get_param(slg,'FaultC')),'GroundFault',char(get_param(slg,'GroundFault')),'AllOff',0);
    r.AllOff = strcmpi(r.AorSwitchA,'off') && strcmpi(r.BorSwitchB,'off') && strcmpi(r.CorSwitchC,'off') && strcmpi(r.GroundFault,'off');
    ok = ok && r.AllOff; rows = [rows; r]; %#ok<AGROW>
    brk = sprintf('%s/LoadSwitch%d', modelName, bus);
    set_param(brk, 'SwitchA', 'off');
    set_param(brk, 'SwitchB', 'off');
    set_param(brk, 'SwitchC', 'off');
    r = struct('BlockType','LoadSwitch','BlockPath',brk,'AorSwitchA',char(get_param(brk,'SwitchA')),'BorSwitchB',char(get_param(brk,'SwitchB')),'CorSwitchC',char(get_param(brk,'SwitchC')),'GroundFault','(n/a)','AllOff',0);
    r.AllOff = strcmpi(r.AorSwitchA,'off') && strcmpi(r.BorSwitchB,'off') && strcmpi(r.CorSwitchC,'off');
    ok = ok && r.AllOff; rows = [rows; r]; %#ok<AGROW>
end
writetable(struct2table(rows), outCsv);
end
