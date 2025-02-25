
% VERSION FOR SCIENCECLOUD ANALYSIS

%% this version implements flexible time window control
addpath(genpath('../canonical-correlation-maps-main')); 
addpath(genpath('../workflow'));
addpath(genpath('../multiarea_analysis'));
addpath(genpath('.'))
startup;

%%
clear all; %clc;
warning off;
setup_colors; 
datasheet = get_data_sheet('multiarea');

%%
dataset = [131:256, 388:473, 475:672];
% dataset = [131:206];
% dataset = dataset(end:-1:1); 
num_trial_type = 4;
tnum = num_trial_type/2;
result_name = 'pca_timewise_learning_sess_120_resampled';

num_neuron_file = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea\results\num_neuron_expert.mat';
num_neuron_expert= load(num_neuron_file);
num_neuron_expert = num_neuron_expert.num_neuron_expert;

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
num_rep = 50;
num_pc = 5;
for dataid = dataset
    
%% load data
dinfo = data_info(datasheet, dataid, 'multiarea', opts.base_dir);
if dinfo.quality_idx==0; continue; end
fprintf('processing dataset %d...\n',dataid);

spath = fullfile(dinfo.work_dir, opts.result_dir);
eid = get_exp_condition_idx(dinfo);
if eid>4; continue; end

data = load_data(dinfo, var_to_read, opts);

if strcmp(dinfo.areas, 'S1/RL')
    area_idx = [1,2];
elseif strcmp(dinfo.areas, 'S1/A')
    area_idx = [1,3];
end


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


%% remove mismatch trials
keep_idx = data.trial_vec<=4;
for a = 1:2
    S_data{a} = S_data{a}(:,keep_idx,:);
end
data.trial_vec = data.trial_vec(keep_idx);
data.num_trial = sum(keep_idx);

for k = 1:length(label)
    label{k} = label{k}(keep_idx);
end

%% remove early licks 
% for a = 1:2
%     for i = 1:data.num_trial
%         lick_frame = data.first_correct_lick(i);
%         if isnan(lick_frame); continue; end
%         idx_remove = lick_frame:ts{3}(end);
%         S_data{a}(:,i,idx_remove) = NaN;
%     end
% end

%% align data to standard
for a = 1:2
    S_data{a} = align_trial_to_standard(S_data{a}, ts, standard_ts);
end

data.trial_length = trial_len;

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

%% repeat the random sample procedure
reverse_str = '';
for rep_idx = 11:num_rep

result_file = fullfile(spath, sprintf('%s_rep_%d.mat', result_name, rep_idx));
% if exist(result_file); continue; end

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


%% timewise pca on neuron dimension
S_pc = cell(num_sess, 2);
num_pc_neuron = nan(num_sess, trial_len, 2);
label_sess = cell(num_sess,length(label));
for m = 1:num_sess
    for a = 1:2
        S = S_data{a}(n_idx{m,a},sess_idx{m},:);  S(isnan(S)) = 0;
        N = num_neuron(m,a);
        npc = nan(1,trial_len);
        S_pc{m,a} = nan(length(sess_idx{m}),trial_len,num_pc);
        for t = 1:trial_len
            St = S;
            v = permute(St(:,:,t), [2,1]);
%             v = v(~isnan(v(:,1)),:);
            [~,sc,~,~,explained] = pca(v);
            explained = cumsum(explained);
            if all(isnan(explained)); continue; end
            npc(t) = find(explained>explained_thr, 1);

            if size(sc,2)<num_pc
                sc = padarray(sc, [0, num_pc-size(sc,2)], NaN, 'post');
            end
            S_pc{m,a}(:,t,:) = sc(:,1:num_pc);

        end
        num_pc_neuron(m,:,a) = npc/num_neuron(m,a) * 100;


        for k = 1:length(label)
            label_sess{m,k} = label{k}(sess_idx{m});
        end

    end
end

%% pca on time dimension
num_pc_time = nan(num_sess, trial_len, 2);
for m = 1:num_sess
    N = length(sess_idx{m});
    for a = 1:2 
        for t = 1:trial_len
            v = S_data{a}(n_idx{m,a},sess_idx{m},t);  v(isnan(v)) = 0;
            [~,~,~,~,explained] = pca(v);

            explained = cumsum(explained);
            if all(isnan(explained)); continue; end
            num_pc_time(m,t,a) = find(explained>explained_thr, 1) / N;
        end
    end
end

%% AUC per PC
num_cond = length(label);
pc_auc = nan(num_sess, trial_len, num_cond, 2, num_pc);
for m = 1:num_sess
    for k = 1:num_cond
        Y = label_sess{m,k};
        for a = 1:2
            for i = 1:num_pc
                X = S_pc{m,a}(:,:,i);
                X(isnan(X)) = 0;
                parfor t = 1:trial_len
                    x = X(:,t);
                    try
                        [~,~,~,pc_auc(m,t,k,a,i)] = perfcurve(Y,x,1);
                    catch ME
                        continue;
                    end
                end
            end
        end
    end
end

%% save
save(result_file, 'beh_rate', 'sess_idx', 'num_sess',...
    'num_pc_neuron','num_pc_time', 'num_neuron', 'pc_auc', '-v7.3');

end

end

