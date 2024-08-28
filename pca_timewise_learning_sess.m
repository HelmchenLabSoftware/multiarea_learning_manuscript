
addpath('../workflow')
addpath(genpath('./'));

%%
clear all; %clc;
warning off;
setup_colors; 
datasheet = get_data_sheet('multiarea');

%%
dataset = [131:256, 388:473, 475:672];
% dataset = [131:206];
dataset = dataset(end:-1:1); 
num_trial_type = 4;
tnum = num_trial_type/2;
result_name = 'pca_timewise_learning_sess_120';

sess_len = 120; min_sess_len = 5;
explained_thr = 70;

opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
% opts.base_dir = '/home/ubuntu/neurophys/Han/data/multiarea';
opts.data_dir = 'data_suite2p'; 
opts.result_dir = 'results_suite2p';
var_to_read = {'S_trial', 'trial_vec', 'num_neuron', 'ts_fr', 'trial'...
    'trial_length', 'ts', 'rate', 'num_ts', 'tvec', 'ts_str', ...
    'choice_time', 'F0', 'task_label', 'num_trial', 'first_correct_lick'};

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

%% remove mismatch trials
keep_idx = data.trial_vec<=4;
for a = 1:2
    S_data{a} = S_data{a}(:,keep_idx,:);
end
data.trial_vec = data.trial_vec(keep_idx);
data.num_trial = sum(keep_idx);

%% split sessions
sess_idx = split_session(data.num_trial, sess_len, min_sess_len);
num_sess = length(sess_idx);

% behavior rate for each session
beh_rate = zeros(1,num_sess);
for n = 1:num_sess
    n_act = sum(data.trial_vec(sess_idx{n})==1) + sum(data.trial_vec(sess_idx{n})==2) ...
        + sum(data.trial_vec(sess_idx{n})==3) + sum(data.trial_vec(sess_idx{n})==4);
    n_correct = sum(data.trial_vec(sess_idx{n})==1) + sum(data.trial_vec(sess_idx{n})==2);
    beh_rate(n) = n_correct/n_act*100;
end

%% align data to standard
for a = 1:2
    S_data{a} = align_trial_to_standard(S_data{a}, ts, standard_ts);
end

data.trial_length = trial_len;

%% timewise pca on neuron dimension
num_pc_neuron = nan(num_sess, trial_len, 2);
for m = 1:num_sess
    for a = 1:2
        S = S_data{a}(:,sess_idx{m},:);  S(isnan(S)) = 0;
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
        num_pc_neuron(m,:,a) = npc/data.num_neuron(a) * 100;
    end
end

%% pca on time dimension
num_pc_time = nan(num_sess, trial_len, 2);
for m = 1:num_sess
    N = length(sess_idx{m});
    for a = 1:2 
        for t = 1:trial_len
            v = S_data{a}(:,sess_idx{m},t);  v(isnan(v)) = 0;
            [~,~,~,~,explained] = pca(v);
            
            explained = cumsum(explained);
            if all(isnan(explained)); continue; end
            num_pc_time(m,t,a) = find(explained>explained_thr, 1) / N;
        end
    end
end


%% save
save(result_file, 'beh_rate', 'sess_idx', 'num_sess',...
    'num_pc_neuron','num_pc_time', '-v7.3');

end

