
addpath('../workflow')
addpath(genpath('./'));

%%
clear all; %clc;
warning off;
setup_colors; 
  
dataset = [131:256, 388:473, 475:672];
% dataset = [131:206];
dataset = dataset(end:-1:1); 
num_trial_type = 4;
tnum = num_trial_type/2;
result_name = 'pca_timewise_learning_tt_match_50';

min_trial = 50;
num_rep = 10;

explained_thr = 70;

opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
% opts.base_dir = '/home/ubuntu/neurophys/Han/data/multiarea';
opts.data_dir = 'data_suite2p'; 
opts.result_dir = 'results_suite2p';
var_to_read = {'S_trial', 'trial_vec', 'num_neuron', 'ts_fr', 'trial'...
    'trial_length', 'ts', 'rate', 'num_ts', 'tvec', 'ts_str', ...
    'choice_time', 'F0', 'task_label', 'num_trial', 'first_correct_lick'};
datasheet = get_data_sheet('multiarea');

standard_trial_structure = [0.5,1,1,0.5,2,4];
standard_fr = 10;
standard_ts = make_trial_structure(standard_trial_structure, standard_fr);
standard_ts_fr = cellfun(@(x) x/standard_fr, standard_ts, 'uniformoutput', false);
num_ts = 6;
trial_len = standard_ts{end}(end);
tvec = (1:trial_len)/standard_fr;

%%
for dataid = dataset
    
%% load data
dinfo = data_info(datasheet, dataid, 'multiarea', opts.base_dir);
if dinfo.quality_idx==0; continue; end
fprintf('processing dataset %d...\n',dataid);

spath = fullfile(dinfo.work_dir, opts.result_dir);
result_file = fullfile(spath, [result_name '.mat']);
eid = get_exp_condition_idx(dinfo);
% if ~exist(result_file); continue; end

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

%% align data to standard
for a = 1:2
    S_data{a} = align_trial_to_standard(S_data{a}, ts, standard_ts);
end

data.trial_length = trial_len;

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

%% repetitions of random sample
num_pc_neuron_tt = nan(num_sess, trial_len, tnum, 2, num_rep);
num_pc_time_tt = nan(num_sess, trial_len, tnum, 2, num_rep);
for rep_idx = 1:num_rep
    
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
    for m = 1:num_sess
        for a = 1:2
            for n = 1:tnum
                S = S_data{a}(:,sess_idx{m},:);  S(isnan(S)) = 0;
                S = S(:,t_idx{m,n},:);
                N = data.num_neuron(a);
                npc = nan(1,trial_len);
                parfor t = 1:trial_len
                    St = S;
                    v = permute(St(:,:,t), [2,1]);
                    [~,~,~,~,explained] = pca(v);
                    explained = cumsum(explained);
                    if all(isnan(explained)); continue; end
                    npc(t) = find(explained>explained_thr, 1);
                end
                num_pc_neuron_tt(m,:,n,a,rep_idx) = npc/data.num_neuron(a) * 100;
            end
        end
    end

    %% pca on time dimension by trial type
    for m = 1:num_sess
        for a = 1:2
            for n = 1:tnum
                for t = 1:trial_len
                    v = S_data{a}(:,sess_idx{m},t);  v(isnan(v)) = 0;
                    v = v(:,t_idx{m,n});
                    [~,~,~,~,explained] = pca(v);
                    explained = cumsum(explained);
                    if all(isnan(explained)); continue; end
                    num_pc_time_tt(m,t,n,a,rep_idx) = find(explained>explained_thr, 1);
                    num_pc_time_tt(m,t,n,a,rep_idx) = num_pc_time_tt(m,t,n,a,rep_idx)/size(v,2)*100;
                end
            end
        end
    end 
    
end

%% save
save(result_file, 'beh_rate', 'sess_idx', 'num_sess', 't_idx', ...
    'num_pc_neuron_tt', 'num_pc_time_tt', ...
    '-v7.3');

end

