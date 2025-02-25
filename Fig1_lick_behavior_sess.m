
clear all;
setup_colors;
datasheet = get_data_sheet('multiarea');

%%
plot_results = 0;
trial_str_combined = {'Correct','Incorrect'};
num_trial_type = 4;
tnum = num_trial_type/2;
dataset = [131:256, 389:672];

opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p';
result_name = 'lick_behavior_sess_2';

%%
for dataid = dataset
    
fprintf('dataset %d...\n',dataid);
dinfo = data_info(datasheet, dataid, 'multiarea', opts.base_dir);

data_dir = dinfo.data_dir;
spath = fullfile(dinfo.work_dir, opts.result_dir);
dpath = fullfile(dinfo.work_dir, opts.data_dir);
fr = dinfo.fr;
num_run = length(data_dir);

%%
if ~exist(spath,'dir')
    mkdir(spath);
end

%% find log files
log_files = cell(size(data_dir));
for k = 1:length(data_dir)
    fnames = dir(fullfile(data_dir{k}, 'behavior', '*.txt'));
    log_files{k} = fullfile(fnames(1).folder, fnames(1).name);
end

%% lick files
lick_files = {};
t = 0;
for k = 1:length(data_dir)
    lick_data_path = fullfile(data_dir{k}, 'behavior', 'lick_data');
    if ~exist(lick_data_path, 'dir')
        [~,log_name] = fileparts(log_files{k});
        lick_data_path = fullfile(data_dir{k}, 'behavior', [log_name '_lick_data']);
    end
    fnames = dir(fullfile(lick_data_path, '*.txt'));
    for i = 1:length(fnames)
        t = t+1;
        lick_files{t} = fullfile(fnames(i).folder,fnames(i).name);
    end
end

%% load trial data  **** note lick files
[trial, rate, offset, lick, trial_structure, ts_str, trial_ts] = ...
    load_behavior_data_2AFC(log_files, lick_files);
trial_vec = make_trial_vec(trial, dinfo.experiment);
num_trial = length(trial_vec);
labels = make_task_labels(trial, dinfo);

% for the new annotation scheme
trial_vec = floor(trial_vec);

%% keep only matched trials
keep_idx = find(trial_vec<=4);
num_trial = length(keep_idx);

%% split session
sess_len = 120;
min_sess_len = 5;
sess_idx = split_session(num_trial, sess_len, min_sess_len);
num_sess = length(sess_idx);

for m = 1:num_sess
    sess_idx{m} = keep_idx(sess_idx{m});
end

%% load lick data
data = load_data(dinfo, {'ts_fr'}, opts);
imaging_ts = cell2mat(data.ts_fr(2:4)');
imaging_ts = imaging_ts - imaging_ts(1);
resample_rate = 30;
resample_ts = 0:1/resample_rate:imaging_ts(end);

t = 0;
lick_length = 20;
lickt = zeros(0,2);
lick1 = {}; lick2 = {};
lick_rate_cue = zeros(0,2);  lick_rate_tex = zeros(0,2);
lick_time_cue = zeros(0,2);  lick_time_tex = zeros(0,2);
for k = 1:length(data_dir)
    lick_data_path = fullfile(data_dir{k}, 'behavior', 'lick_data');
    if ~exist(lick_data_path, 'dir')
        [~,log_name] = fileparts(log_files{k});
        lick_data_path = fullfile(data_dir{k}, 'behavior', [log_name '_lick_data']);
    end
    fnames = dir(fullfile(lick_data_path, '*.txt'));
    % load separate lick TTL files
    for i = 1:length(fnames)
        t = t+1;
        lick_file = dlmread(fullfile(fnames(i).folder,fnames(i).name), '\t');
        lick_file = lick_file(2:end,:);
        
        % downsample lick ttl data
        lick1{t} = interp1(lick_file(:,1), lick_file(:,2), resample_ts);
        lick2{t} = interp1(lick_file(:,1), lick_file(:,3), resample_ts);
        
        % correct for multiple licks
        lick1_time = lick_file(split_licks(lick_file(:,2), lick_length),1);
        lick2_time = lick_file(split_licks(lick_file(:,3), lick_length),1);
        
        % first lick of the last lick bout
        if isempty(lick1_time); t1 = NaN; end
        if isempty(lick2_time); t2 = NaN; end
        t_all = [lick1_time; lick2_time];
        if ~isempty(t_all)
            t_all = sort(t_all, 'ascend'); t_port = ones(size(t_all));
            for n = 1:length(t_all)
                if any(lick2_time==t_all(n)); t_port(n) = 2; end
            end
            if all(t_port==1);  t1 = lick1_time(1); end
            if all(t_port==2);  t2 = lick2_time(1); end
            if length(unique(t_port))==2  % switch lick
                idx = find(diff(t_port)==-1); 
                if ~isempty(idx)
                    idx = idx(end); t1 = t_all(idx+1);
                else  % switched only once
                    t1 = lick1_time(1);
                end
                idx = find(diff(t_port)==1);
                if ~isempty(idx)
                    idx = idx(end); t2 = t_all(idx+1);
                else  % switched only once
                    t2 = lick2_time(1);
                end
            end
        end
        % first ever lick on this port
        lickt(t,:) = [t1,t2];
        
        % lick rate
        lick_rate_cue(t,1) = sum(lick1_time<=trial_ts.cue(i))/trial_ts.cue(i);
        lick_rate_cue(t,2) = sum(lick2_time<=trial_ts.cue(i))/trial_ts.cue(i);
        lick_rate_tex(t,1) = sum(lick1_time>trial_ts.cue(i))/trial_ts.present(i);
        lick_rate_tex(t,2) = sum(lick2_time>trial_ts.cue(i))/trial_ts.present(i);

        % lick time
        t_idx = lick_file(:,1)<=trial_ts.cue(i);
        lick_time_cue(t,1) = sum(lick_file(t_idx,2)==1)/length(t_idx);
        lick_time_cue(t,2) = sum(lick_file(t_idx,3)==1)/length(t_idx);
        t_idx = lick_file(:,1)>trial_ts.cue(i);
        lick_time_tex(t,1) = sum(lick_file(t_idx,2)==1)/length(t_idx);
        lick_time_tex(t,2) = sum(lick_file(t_idx,3)==1)/length(t_idx);

    end
end

%% process lick binary result
maxt = max(cellfun(@(x) length(x), cat(2,lick1,lick2)));
lick1 = cellfun(@(x) padarray(x, [maxt-length(x),0], NaN, 'post'), lick1, 'uniformoutput', 0);
lick2 = cellfun(@(x) padarray(x, [maxt-length(x),0], NaN, 'post'), lick2, 'uniformoutput', 0);
lick1 = cell2mat(lick1');
lick2 = cell2mat(lick2');
lick_port_rate = cell(num_sess,num_trial_type,2);
for m = 1:num_sess
    for n = 1:num_trial_type
        lick_port_rate{m,n,1} = lick1(sess_idx{m},:);
        lick_port_rate{m,n,1} = lick_port_rate{m,n,1}(trial_vec(sess_idx{m})==n,:);
        lick_port_rate{m,n,2} = lick2(sess_idx{m},:);
        lick_port_rate{m,n,2} = lick_port_rate{m,n,2}(trial_vec(sess_idx{m})==n,:);
    end
end

%% pull results
beh_rate = zeros(num_sess,2,3);
beh_rate_combined = zeros(num_sess,1,3);
lick_rate = struct('cue',{cell(num_sess,2,num_trial_type)},'texture',{cell(num_sess,2,num_trial_type)},...
            'cue_prev',{cell(num_sess,tnum,2)},'texture_prev',{cell(num_sess,tnum,2)},...
            'cue_next',{cell(num_sess,tnum,2)},'texture_next',{cell(num_sess,tnum,2)});
lick_time = struct('cue',{cell(num_sess,2,num_trial_type)},'texture',{cell(num_sess,2,num_trial_type)},...
            'cue_prev',{cell(num_sess,tnum,2)},'texture_prev',{cell(num_sess,tnum,2)},...
            'cue_next',{cell(num_sess,tnum,2)},'texture_next',{cell(num_sess,tnum,2)});
lick_first = cell(num_sess,num_trial_type);


for m = 1:num_sess
    
    num_cue1 = length(intersect(sess_idx{m}, trial.cue1));
    num_cue2 = length(intersect(sess_idx{m}, trial.cue2));
    num_cue3 = length(intersect(sess_idx{m}, trial.cue3));
    num_cue4 = length(intersect(sess_idx{m}, trial.cue4));
    num_correct1 = length(intersect(sess_idx{m}, trial.hit1));
    num_correct2 = length(intersect(sess_idx{m}, trial.hit2));
    num_incorrect1 = length(intersect(sess_idx{m}, trial.FA1));
    num_incorrect2 = length(intersect(sess_idx{m}, trial.FA2));
    num_miss1 = length(intersect(sess_idx{m}, trial.miss1));
    num_miss2 = length(intersect(sess_idx{m}, trial.miss2));

    num_cue1_choice = num_correct1 + num_incorrect1;
    num_cue2_choice = num_correct2 + num_incorrect2;

    % normal trials
    beh_rate(m,1,1) = num_correct1/num_cue1_choice;
    beh_rate(m,1,2) = num_incorrect1/num_cue1_choice;
    beh_rate(m,1,3) = num_miss1/num_cue1;
    beh_rate(m,2,1) = num_correct2/num_cue2_choice;
    beh_rate(m,2,2) = num_incorrect2/num_cue2_choice;
    beh_rate(m,2,3) = num_miss2/num_cue2;
    beh_rate_combined(m,1) = (num_correct1+num_correct2)/(num_cue1_choice+num_cue2_choice);
    beh_rate_combined(m,2) = (num_incorrect1+num_incorrect2)/(num_cue1_choice+num_cue2_choice);
    beh_rate_combined(m,3) = (num_miss1+num_miss2)/(num_cue1+num_cue2);

    %% lick data
    trial_vec_sess = trial_vec(sess_idx{m});
    for n = 1:num_trial_type
        t_idx = trial_vec_sess==n;

        % lick rate
        lick_rate.cue{m,1,n} = lick_rate_cue(sess_idx{m}(t_idx),1);
        lick_rate.cue{m,2,n} = lick_rate_cue(sess_idx{m}(t_idx),2);
        lick_rate.texture{m,1,n} = lick_rate_tex(sess_idx{m}(t_idx),1);
        lick_rate.texture{m,2,n} = lick_rate_tex(sess_idx{m}(t_idx),2);

        % lick time
        lick_time.cue{m,1,n} = lick_time_cue(sess_idx{m}(t_idx),1);
        lick_time.cue{m,2,n} = lick_time_cue(sess_idx{m}(t_idx),2);
        lick_time.texture{m,1,n} = lick_time_tex(sess_idx{m}(t_idx),1);
        lick_time.texture{m,2,n} = lick_time_tex(sess_idx{m}(t_idx),2);

    end

    % first lick time
    choice_sess = labels.choice_vec(sess_idx{m});
    lick_sess = lickt(sess_idx{m},:);
    for n = 1:num_trial_type
        idx = find(trial_vec_sess==n);
        if ~isempty(idx) && choice_sess(idx(1))~=0 && ~isempty(lick_sess)
            lick_first{m,n} = lick_sess(idx, choice_sess(idx(1)));
        end
    end


end


%% save results
save(fullfile(spath, [result_name '.mat']),'beh_rate', 'beh_rate_combined',...
    'lick_first','lick_rate','lick_time',...
    'lick_port_rate', 'resample_ts', 'trial_vec', 'trial', 'sess_idx', '-v7.3');

end
