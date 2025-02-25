
addpath('../calcium_analysis');
addpath(genpath('./'));
addpath(genpath('../canonical-correlation-maps-main')); 

%% this version split sessions to ensure minimum number of trials each trial type
clear;
load('mycc.mat');
setup_colors;

warning off; 
 
%% dataset information
opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
% opts.base_dir = '/home/ubuntu/neurophys/data/multiarea'; 
opts.data_dir = 'data_suite2p'; 
opts.result_dir = 'results_suite2p'; 
var_to_read = {'trial_vec', 'num_neuron', 'num_trial', 'trial_length', 'S_trial',  ...
    'ts', 'choice_time', 'task_label', 'F0', 'first_correct_lick'};
 
dataset = [131:256, 388:672];
% dataset = [131:256, 388:529];
dataset = dataset(end:-1:1);
datasheet = get_data_sheet('multiarea');
result_name = 'taskvar_vs_cca_axis_sess_pc30_tt_50_match_decoder_auc';

min_trial = 50;

% shuffle_method = 'frame';
shuffle_method = 'trial';
% shuffle_method = 'all';

sig_dim_thr = 1.96;
% sig_dim_thr = 3;

weighted_sim = 1;

sim_shuff_method = 'model';
% sim_shuff_method = 'perm';
% sim_shuff_method = 'const';
% var_thr = 70;

max_pc = 30; 
num_rep = 10;


num_shuff = 100;
num_shuff2 = 50;
num_ts = 6;
tnum = 2;
plot_result = 0;

%%
for dataid = dataset
    
%% load data
dinfo = data_info(datasheet, dataid, 'multiarea', opts.base_dir);
spath = fullfile(dinfo.work_dir, opts.result_dir);
eid = get_exp_condition_idx(dinfo);
if dinfo.quality_idx==0; continue; end
if dataid==474; continue; end
if eid>=5; continue; end

fprintf('processing dataset %d...\n',dataid);
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
if isempty(ts{1}); ts{1} = 1; end
ts{4} = ts{3}(end)+1:ts{3}(end)+choice_tw;
ts{5} = (ts{4}(end)+1) : (ts{4}(end)+1) + length(ts{5});
ts{6} = (ts{5}(end)+1) : size(S_data{1},3);
data.trial_length = ts{6}(end);

%% remove early licks 
for a = 1:2
    for i = 1:data.num_trial
        lick_frame = data.first_correct_lick(i);
        if isnan(lick_frame); continue; end
        idx_remove = lick_frame:ts{3}(end);
        S_data{a}(:,i,idx_remove) = NaN;
    end
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
    if any(cellfun(@(x) length(x), trial_idx)<5)
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

%% repeat the random split many times, save to different files for now
for rep_idx = 1:num_rep
    
    tic;
    result_file = fullfile(spath, sprintf('%s_rep_%d.mat', result_name, rep_idx));
    % if exist(result_file); continue; end

    
    %% bootstrap each trial type to match trial numbers
    t_idx_sess = cell(0, tnum);
    t_idx_sess_idx = cell(num_sess,1);
    for m = 1:num_sess
        N = min([length(t_idx_sess_full{m,1}), length(t_idx_sess_full{m,2})]);
        for tt = 1:tnum
            rand_idx = randperm(length(t_idx_sess_full{m,tt}));
            rand_idx = rand_idx(1:N);
            t_idx_sess{m,tt} = t_idx_sess_full{m,tt}(rand_idx);
            t_idx_sess_idx{m}(end+1:end+N) = tt;
        end
    end

    %% split within area subsets
    idx_sub = cell(num_sess,2,2);
    N_sub = zeros(num_sess,2,2);

    for m = 1:num_sess
        for a = 1:2
            N = data.num_neuron(a);
            rand_idx = randperm(N);
            N_sub(m,a,1) = round(N/2);
            N_sub(m,a,2) = N-N_sub(m,a,1);
            idx_sub{m,a,1} = rand_idx(1:N_sub(m,a,1));
            idx_sub{m,a,2} = rand_idx(N_sub(m,a,1)+1:N_sub(m,a,1)+N_sub(m,a,2));
        end
    end

    %% pca
    pc_coef = cell(num_sess,2,2);
    data.S_pc = cell(num_sess,2,2);
    S_num = zeros(num_sess,2,2);
    var_explained = zeros(num_sess,2,2);
    pca_weight = cell(num_sess,2,2);
    for m = 1:num_sess
        for a = 1:2
            for i = 1:2
                v0 = S_data{a}(idx_sub{m,a,i},sess_idx{m},:);
                v0 = v0(:,cell2mat(t_idx_sess(m,:)),:);
                v = reshape(permute(v0,[3,2,1]), [], N_sub(m,a,i));
%                 v = zscore_nan(v, [], 1);
                v(isnan(v)) = 0;
                [pc_coef{m,a,i}, ~, ~, ~, explained_raw, mu] = pca(v);
                explained = cumsum(explained_raw);
                % S_num(m,a,i) = find(explained>var_thr, 1);  % define by explained variance
                S_num(m,a,i) = min(max_pc, length(explained));
                var_explained(m,a,i) = explained(S_num(m,a,i));
                pca_weight{m,a,i} = explained_raw(1:S_num(m,a,i))'/100;
                pc_coef{m,a,i} = pc_coef{m,a,i}(:,1:S_num(m,a,i));
                for t = 1:data.trial_length
                    v = v0(:,:,t); 
%                     v(isnan(v)) = 0;
                    sc = (v - mu'*ones(1,size(v0,2)))'*pc_coef{m,a,i};
                    data.S_pc{m,a,i}(:,:,t) = sc';
                end
            end
        end
    end
    S_sess = data.S_pc;

    %% task labels
    label = {data.task_label.cue_vec, ...
            data.task_label.tex_vec, ...
            data.task_label.choice_vec, ...
            data.task_label.rew_vec};
    label_str = {'tone', 'texture', 'choice', 'reward'};
    decoder_tw = {ts{2}, ts{3}, ts{4}, ts{5}};

    trial_vec_sess = cell(num_sess,1);
    label_sess = cell(num_sess,length(label));
    for m = 1:num_sess
        trial_vec_sess{m} = data.trial_vec(sess_idx{m});
        for k = 1:length(label)
            label_sess{m,k} = label{k}(sess_idx{m});
            label_sess{m,k} = label_sess{m,k}(cell2mat(t_idx_sess(m,:)));
        end
    end
    K = length(label);

    %% compute task variable encoding direction 
    pred_auc = nan(num_sess, K, tnum, 2, 2);
    pred_auc_shuff = nan(num_sess, K, tnum, 2, num_shuff, 2);
    task_axis = cell(num_sess, K, tnum, 2, 2);
    for k = 1:length(label)
        tw = decoder_tw{k};
        for m = 1:num_sess
            sess_N = length(sess_idx{m});
            for a = 1:2
                for sub_idx = 1:2
                    for tt = 1:tnum
                        
                        X = S_sess{m,a,sub_idx};
                        Y = label_sess{m,k};

                        % take only correct trials
                        X = X(:,t_idx_sess_idx{m}==tt,:);
                        Y = Y(t_idx_sess_idx{m}==tt);

                        N = size(X,1);
                        if sum(Y==1)==0 || sum(Y==2)==0
                            task_axis{m,k,tt,a,sub_idx} = nan(S_num(m,a),1);
                        end
                        if isempty(X)
                            task_axis{m,k,tt,a,sub_idx} = nan(S_num(m,a),1);
                        end

                        v1 = nanmean(nanmean(X(:,Y==1,tw),2),3);
                        v2 = nanmean(nanmean(X(:,Y==2,tw),2),3);
                        b = (v2-v1)/2;
                        task_axis{m,k,tt,a,sub_idx} = b;

                        %% projection and auc
%                         b = task_axis{m,k,tt,a,sub_idx};
                        b = task_axis{m,k,1,a,sub_idx};
                        Xn = nanmean(X(:,:,tw), 3);
                        pred = Xn'*b;

                        %% shuffled control
                        pred_shuff = nan(num_shuff, size(X,2));
                        for s = 1:num_shuff
                            Xs = nanmean(X(randperm(N),:,tw), 3);
                            pred_shuff(s,:) = Xs'*b;
                        end

                        %% prediction auc
                        x = pred';  y = Y';

                        try; [~,~,~,pred_auc(m,k,tt,a,sub_idx)] = perfcurve(y, x, 2);
                        end
                        % shuffled
                        ps = nan(1, num_shuff);
                        parfor s = 1:num_shuff
                            x = pred_shuff(s,:)'; y = Y';
                            try; [~,~,~,ps(s)] = perfcurve(y, x, 2);
                            end
                        end
                        pred_auc_shuff(m,k,tt,a,:,sub_idx) = ps;
                        
                    end
                    
                end
            end
        end
    end
    
    %% decoder similarity
    svm_sim = nan(num_sess, K, K, tnum, 2, 2);
    for m = 1:num_sess
        for sub_idx = 1:2
            for a = 1:2
                for tt = 1:tnum
                    b = cell2mat(task_axis(m,:,tt,a,sub_idx))';
                    if weighted_sim  % weighted similarity
                        w = pca_weight{m,a,sub_idx};
                        dist_fun = @(u,v) (w.*u*v') ./ (sqrt(sum(w.*(u.^2))) * sqrt(w*(v.^2)'));
                        svm_sim(m,:,:,tt,a,sub_idx) = pdist2(b,b,dist_fun);
                    else
                        svm_sim(m,:,:,tt,a,sub_idx) = 1-pdist2(b, b, 'cos');
                    end
                end
            end
        end
    end

    
    %% save
    save(result_file, 'beh_rate', 'sess_idx', 'num_sess', 'S_num',...
        'var_explained', 'task_axis', 't_idx_sess', ...
        'pred_auc', 'pred_auc_shuff', 'svm_sim',...
        '-v7.3');
    toc;

end
    


    
end



