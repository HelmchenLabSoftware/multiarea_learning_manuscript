 %%
addpath('../workflow')
addpath(genpath('./'));

%%
clear;
load('mycc.mat');
setup_colors;
datasheet = get_data_sheet('multiarea');
 
warning off;

%% dataset information
opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p';
var_to_read = {'trial_vec', 'num_neuron', 'num_trial', 'trial_length', 'S_trial',  ...
    'ts', 'choice_time', 'task_label', 'F0', 'first_correct_lick'};
result_name = 'top_pc_auc_sess_2';
 
dataset = [131:186,188:256, 389:672]; 
% dataset = dataset(end:-1:1); 

sess_len = 120; min_sess_len = 5;

standard_trial_structure = [0.5,1,1,0.5,2,4];
standard_fr = 10;
standard_ts = make_trial_structure(standard_trial_structure, standard_fr);
standard_ts_fr = cellfun(@(x) x/standard_fr, standard_ts, 'uniformoutput', false);
trial_len = standard_ts{end}(end);
tvec = (1:trial_len)/standard_fr;

num_shuff = 10;
num_ts = 6;
tnum = 2;
plot_result = 0;
 
%%
for dataid = dataset
    
%% dataset information
dinfo = data_info(datasheet, dataid, 'multiarea', opts.base_dir);
if dinfo.quality_idx==0; continue; end
eid = get_exp_condition_idx(dinfo);
if eid>=5; continue; end

spath = fullfile(dinfo.work_dir, opts.result_dir);
result_file = fullfile(spath, [result_name '.mat']);
% if exist(result_file); continue; end

fprintf('processing dataset %d\n', dataid);

%% assemble data from trials
data = load_data(dinfo, var_to_read, opts);

% for dataset 242, choice window is empty
if isempty(data.ts{4}); data.ts{4} = data.ts{3}(end):data.ts{5}(1); end
if data.ts{2}(1)==0; data.ts{2} = data.ts{2}(2:end); end

%% task labels
data.task_label.err_vec = data.task_label.rew_vec;
data.task_label.err_vec(data.task_label.err_vec>0) = 1;
data.task_label.err_vec(data.task_label.err_vec==0) = 2;
label = {data.task_label.cue_vec, ...
        data.task_label.tex_vec, ...
        data.task_label.choice_vec, ...
        data.task_label.rew_vec};

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
for n = 1:length(label)
    label{n} = label{n}(keep_idx);
end
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
num_neuron = data.num_neuron;
num_trial = data.num_trial;
trial_vec = data.trial_vec;

%% split sessions
sess_idx = split_session(data.num_trial, sess_len, min_sess_len);
num_sess = length(sess_idx);

% behavior rate for each session
beh_rate = zeros(1,num_sess);
miss_rate = zeros(1,num_sess);
for n = 1:num_sess
    n_act = sum(data.trial_vec(sess_idx{n})==1) + sum(data.trial_vec(sess_idx{n})==2) ...
        + sum(data.trial_vec(sess_idx{n})==3) + sum(data.trial_vec(sess_idx{n})==4)...
        + sum(data.trial_vec(sess_idx{n})==7.5) + sum(data.trial_vec(sess_idx{n})==8.5)...
        + sum(data.trial_vec(sess_idx{n})==7.6) + sum(data.trial_vec(sess_idx{n})==8.6);
    n_noact = sum(data.trial_vec(sess_idx{n})==5) + sum(data.trial_vec(sess_idx{n})==6) ...
        + sum(data.trial_vec(sess_idx{n})==7) + sum(data.trial_vec(sess_idx{n})==8)...
        + sum(data.trial_vec(sess_idx{n})==13) + sum(data.trial_vec(sess_idx{n})==14);
    n_correct = sum(data.trial_vec(sess_idx{n})==1) + sum(data.trial_vec(sess_idx{n})==2);
    beh_rate(n) = n_correct/n_act*100;
    miss_rate(n) = n_noact/length(sess_idx{n})*100;
end

%% sliding PCA on each sess
num_pc = 5;
S_sess = cell(num_sess,2);
trial_vec_sess = cell(num_sess,1);
label_sess = cell(num_sess,length(label));
for m = 1:num_sess
    trial_vec_sess{m} = data.trial_vec(sess_idx{m});
    for a = 1:2
        S_sess{m,a} = nan(length(sess_idx{m}),trial_len,num_pc);
        for t = 1:trial_len
            v = S_data{a}(:,sess_idx{m},t)';
            v(isnan(v)) = 0;
            [~,sc] = pca(v);
            if size(sc,2)<num_pc
                sc = padarray(sc, [0, num_pc-size(sc,2)], NaN, 'post');
            end
            S_sess{m,a}(:,t,:) = sc(:,1:num_pc);
        end
    end
    for k = 1:length(label)
        label_sess{m,k} = label{k}(sess_idx{m});
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
                X = S_sess{m,a}(:,:,i);
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
                    
%% by trial type
pc_auc_tt = nan(num_sess, trial_len, num_cond, tnum, 2, num_pc);
for m = 1:num_sess
    for k = 1:num_cond
        Y = label_sess{m,k};
        for tt = 1:tnum
            t_idx = trial_vec_sess{m}==2*tt-1 | trial_vec_sess{m}==2*tt;
            y = Y(t_idx);
            for a = 1:2
                for i = 1:num_pc
                    X = S_sess{m,a}(t_idx,:,i);
                    X(isnan(X)) = 0;
                    parfor t = 1:trial_len
                        x = X(:,t);
                        try
                            [~,~,~,pc_auc_tt(m,t,k,tt,a,i)] = perfcurve(y,x,1);
                        catch ME
                            continue;
                        end
                    end
                end
            end
        end
    end
end


%% 
if plot_result

    %% plot
    k = 1;
    figure;
    for i = 1:num_pc
        for a = 1:2
            subplot(num_pc,2,(i-1)*2+a); hold on; 
            v = pc_auc(:,:,k,a,i)';
            v = abs(v*2-1);
            plot(v, 'color', 'k');
%             plot([1 trial_len], [0 0], 'k:');
            ylim([0 1]);
            draw_trial_structure(standard_ts);
            xlim([0 standard_ts{5}(end)]);
        end
    end
    
    
    %% plot by trial type
    k=1;
    figure;
    for i = 1:num_pc
        for a = 1:2
            subplot(num_pc,2,(i-1)*2+a); hold on; 
            for tt = 1:tnum
                v = pc_auc_tt(:,:,k,tt,a,i)';
                v = abs(v*2-1);
                plot(v, 'color', cc_trial{2*tt-1});
            end
            ylim([0 1]);
            draw_trial_structure(standard_ts);
            xlim([0 standard_ts{5}(end)]);
        end
    end

    
end
    
    %% save
    save(result_file, 'beh_rate', 'miss_rate', 'sess_idx', 'num_sess',...
        'pc_auc', 'pc_auc_tt', '-v7.3');

    
end




