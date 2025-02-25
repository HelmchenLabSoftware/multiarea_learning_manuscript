
% VERSION FOR SCIENCECLOUD ANALYSIS

%% this version implements flexible time window control
addpath(genpath('../canonical-correlation-maps-main')); 

%%
clear all; %clc;
warning off;
setup_colors; 
  
num_neuron_file = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea\results\num_neuron_expert.mat';
num_neuron_expert= load(num_neuron_file);
num_neuron_expert = num_neuron_expert.num_neuron_expert;


dataset = [131:256, 388:473, 475:672];
% dataset = [131:256, 388:473, 475:529];
% dataset = dataset(end:-1:1);
num_trial_type = 4;
tnum = num_trial_type/2;
% result_name = 'pca_tw_learning_tt_match_resample_50';
result_name = 'pca_tw_learning_tt_match_resample_50_rep50';

min_trial = 50;
num_rep = 50;
num_pc = 5; 
explained_thr = 70;
num_ts = 6;

opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
% opts.base_dir = '/home/ubuntu/neurophys/Han/data/multiarea';
opts.data_dir = 'data_suite2p'; 
opts.result_dir = 'results_suite2p';
var_to_read = {'S_trial', 'trial_vec', 'num_neuron', 'ts_fr', 'trial'...
    'trial_length', 'ts', 'rate', 'num_ts', 'tvec', 'ts_str', ...
    'choice_time', 'F0', 'task_label', 'num_trial', 'first_correct_lick'};
datasheet = get_data_sheet('multiarea');
 

%%
for dataid = dataset
    
%% load data
% dinfo = data_info_gs(dataid, 'multiarea', opts.base_dir);
dinfo = data_info(datasheet, dataid, 'multiarea', opts.base_dir);

spath = fullfile(dinfo.work_dir, opts.result_dir);
result_file = fullfile(spath, [result_name '.mat']);
eid = get_exp_condition_idx(dinfo);
if dinfo.quality_idx==0; continue; end
if dataid==474; continue; end
if eid>=5; continue; end
% if ~exist(result_file); continue; end
fprintf('processing dataset %d...\n',dataid);

if strcmp(dinfo.areas, 'S1/RL')
    area_idx = [1,2];
elseif strcmp(dinfo.areas, 'S1/A')
    area_idx = [1,3];
end

data = load_data(dinfo, var_to_read, opts);

%% normalize data
for a = 1:2
    data.F0{a}(data.F0{a}==0) = 1;
    data.S_trial{a} = data.S_trial{a}./repmat(data.F0{a}, 1, data.num_trial, data.trial_length); 
end

% zscore each neuron
for a = 1:2
    v = reshape(permute(data.S_trial{a},[3,2,1]), [], data.num_neuron(a));
    v = zscore_nan(v, [], 1);
    data.S_trial{a} = permute(reshape(v, data.trial_length, data.num_trial, data.num_neuron(a)), [3,2,1]);
end

% align data
for a = 1:2
    data.S_rew{a} = align_data_by_choice(data.S_trial{a}, data.choice_time, data.ts);
    data.S_rew_shift{a} = align_data_by_choice_shift(data.S_trial{a}, data.choice_time, data.ts);
end
S_num = data.num_neuron;

%% handle choice window
choice_tw = 5;
S_data = cell(1,2);
ts = data.ts;
for a = 1:2
    S_data{a} = data.S_rew{a};
    S_data{a}(:,:,ts{3}(end)+1:ts{3}(end)+choice_tw) = ...
        data.S_rew_shift{a}(:,:,ts{4}(end)-choice_tw+1:ts{4}(end));
    S_data{a} = S_data{a}(:,:,[1:ts{3}(end)+choice_tw, ts{5}(1):end]);
end
if isempty(ts{1}); ts{1} = 1:2; end
ts{4} = ts{3}(end)+1:ts{3}(end)+choice_tw;
ts{5} = (ts{4}(end)+1) : (ts{4}(end)+1) + length(ts{5});
ts{6} = (ts{5}(end)+1) : size(S_data{1},3);
data.trial_length = ts{6}(end);

%% task labels
data.task_label.err_vec = data.task_label.rew_vec;
data.task_label.err_vec(data.task_label.err_vec>0) = 1;
data.task_label.err_vec(data.task_label.err_vec==0) = 2;
label = {data.task_label.cue_vec, ...
        data.task_label.tex_vec, ...
        data.task_label.choice_vec, ...
        data.task_label.rew_vec};
num_cond = length(label);

label_ts = {ts{2}, ts{3}, ts{4}, ts{5}};

%% remove early licks 
for a = 1:2
    for i = 1:data.num_trial
        lick_frame = data.first_correct_lick(i);
        if isnan(lick_frame); continue; end
        idx_remove = lick_frame:ts{3}(end);
        S_data{a}(:,i,idx_remove) = NaN;
    end
end

%% remove mismatch trials
keep_idx = data.trial_vec<=4;
for a = 1:2
    S_data{a} = S_data{a}(:,keep_idx,:);
end
data.trial_vec = data.trial_vec(keep_idx);
data.num_trial = sum(keep_idx);

for k = 1:num_cond
    label{k} = label{k}(keep_idx);
end

%% split sessions to ensure min. number of trials per trial type
t_idx = cell(1,tnum);
for tt = 1:tnum
    t_idx{tt} = find(data.trial_vec==2*tt-1 | data.trial_vec==2*tt);
end

num_sess = 0;
flag = 0;
idx_last = 0;
sess_idx = {};
beh_rate = zeros(0,num_sess);
t_idx_sess_full = cell(0, tnum);
% assume only two trial types, correct and incorrect
while flag==0
    trial_idx = cell(1,tnum);
    for tt = 1:tnum
        N = min(min_trial, length(t_idx{tt}));
        trial_idx{tt} = t_idx{tt}(1:N);
        t_idx{tt} = t_idx{tt}(N+1:end);
    end
    if any(cellfun(@(x) length(x), trial_idx)==0)
        break;
    end
    
    num_sess = num_sess + 1;
    idx_stop = max(cellfun(@(x) x(end), trial_idx));  % make sure the least trial number
    sess_idx{num_sess} = idx_last+1 : idx_stop;
    % finish when not enough trials are found
    if any(cellfun(@(x) length(x), trial_idx) < min_trial)
        flag = 1;
    end
    idx_last = idx_stop;
    
    % store trial type indices in session
    trial_vec_sess = data.trial_vec(sess_idx{num_sess});
    for tt = 1:tnum
        t_idx_sess_full{num_sess,tt} = find(trial_vec_sess==2*tt-1 | trial_vec_sess==2*tt);
    end
    
    % compute session performance
    beh_rate(num_sess) = (sum(trial_vec_sess==1) + sum(trial_vec_sess==2))/...
        sum(trial_vec_sess<=4) * 100; 
    
end

idx = cellfun(@(x) length(x), sess_idx)>=5;
sess_idx = sess_idx(idx);
t_idx_sess_full = t_idx_sess_full(idx,:);
beh_rate = beh_rate(idx);
num_sess = sum(idx);

%% repetitions
num_pc_neuron_tt = nan(num_sess, num_ts, tnum, 2, num_rep);
num_pc_time_tt = nan(num_sess, num_ts, tnum, 2, num_rep);
pc_auc_tt = nan(num_sess, num_cond, tnum, 2, num_pc, num_rep);

reverse_str = '';
for rep_idx = 1:num_rep
    
    % print current progress
    msg = sprintf('Repetetion done: %d/%d\n', rep_idx, num_rep);
    fprintf([reverse_str, msg]);
    reverse_str = repmat(sprintf('\b'), 1, length(msg));

    %% resample neuron number
    n_idx = cell(num_sess, 2);
    num_neuron = zeros(num_sess, 2);
    for a = 1:2
        for m = 1:num_sess
            if beh_rate(m)>=75
                num_neuron(m,a) = data.num_neuron(a);
                idx = 1:data.num_neuron(a);
            else
                rand_idx = randperm(length(num_neuron_expert{area_idx(a)}), 1);
                N_resample = num_neuron_expert{area_idx(a)}(rand_idx);
                if N_resample < data.num_neuron(a)
                    idx = randperm(data.num_neuron(a), N_resample);
                else
                    idx = 1:data.num_neuron(a);
                end
            end
            n_idx{m,a} = idx;
            num_neuron(m,a) = length(idx);
        end
    end

    %% bootstrap each trial type to match trial numbers
    t_idx = cell(0, tnum);
    for m = 1:num_sess
        N = min([length(t_idx_sess_full{m,1}), length(t_idx_sess_full{m,2})]);
        for tt = 1:tnum
            rand_idx = randperm(length(t_idx_sess_full{m,tt}));
            rand_idx = rand_idx(1:N);
            t_idx{m,tt} = t_idx_sess_full{m,tt}(rand_idx);
        end
    end

    %% timewise pca on neuron dimension by trial type
    S_pc = cell(num_sess, tnum, 2);
    label_sess = cell(num_sess,tnum,length(label));
    for m = 1:num_sess
        for tt = 1:tnum
            for a = 1:2
                S_pc{m,tt,a} = nan(length(t_idx{m,tt}), num_ts, num_pc);
                S = S_data{a}(n_idx{m,a},sess_idx{m},:);
                S = S(:,t_idx{m,tt},:);
                npc = nan(1,num_ts);
                for t = 1:num_ts
                    v = reshape(permute(S(:,:,ts{t}), [3,2,1]), [], num_neuron(m,a));
                    v(isnan(v)) = 0;
                    [~,sc,~,~,explained] = pca(v);
                    explained = cumsum(explained);
                    if all(isnan(explained)); continue; end
                    npc(t) = find(explained>explained_thr, 1);

                    v_pc = reshape(sc, length(ts{t}), length(t_idx{m,tt}), size(sc,2));
                    v_pc = permute(v_pc, [2,1,3]);
                    N = min(num_pc, size(v_pc,3));
                    S_pc{m,tt,a}(:,ts{t},1:N) = v_pc(:,:,1:N);

                end
                num_pc_neuron_tt(m,:,tt,a,rep_idx) = npc/num_neuron(m,a) * 100;
            end
            for k = 1:num_cond
                label_sess{m,tt,k} = label{k}(sess_idx{m});
                label_sess{m,tt,k} = label_sess{m,tt,k}(t_idx{m,tt});
            end

        end
    end

    %% pca on time dimension by trial type
    explained_thr = 70;
    for m = 1:num_sess
        for a = 1:2
            for tt = 1:tnum
                for t = 1:num_ts
                    v = S_data{a}(n_idx{m,a},sess_idx{m},ts{t});
                    v = reshape(permute(v(:,t_idx{m,tt},:), [3,1,2]), [], length(t_idx{m,tt}));
                    v(isnan(v)) = 0;
                    [~,~,~,~,explained] = pca(v);
                    explained = cumsum(explained);
                    if all(isnan(explained)); continue; end
                    num_pc_time_tt(m,t,tt,a) = find(explained>explained_thr, 1);
                end
                num_pc_time_tt(m,:,tt,a,rep_idx) = num_pc_time_tt(m,:,tt,a)/length(t_idx{m,tt}) * 100;
            end
        end
    end

    %% AUC by trial type
    for m = 1:num_sess
        for k = 1:num_cond
            for tt = 1:tnum
                y = label_sess{m,tt,k};
                for a = 1:2
                    for i = 1:num_pc
                        X = S_pc{m,tt,a}(:,:,i);
                        X(isnan(X)) = 0;
                        x = X(:,label_ts{k});
                        x = nanmean(x,2);
                        try;[~,~,~,pc_auc_tt(m,k,tt,a,i,rep_idx)] = perfcurve(y,x,1); end
                    end
                end
            end
        end
    end

end

%% save
save(result_file, 'beh_rate', 'sess_idx', 'num_sess', 't_idx', ...
    'num_pc_neuron_tt', 'num_pc_time_tt', 'pc_auc_tt', 'num_neuron',...
    '-v7.3');

end



