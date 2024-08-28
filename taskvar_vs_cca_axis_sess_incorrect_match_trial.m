
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
% dataset = [452:529];
dataset = dataset(end:-1:1);
datasheet = get_data_sheet('multiarea');
result_name = 'taskvar_vs_cca_axis_sess_pc30_tt_50_match_2_nozs';

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
num_rep = 100;


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
% for a = 1:2
%     v = reshape(permute(data.S_trial{a},[3,2,1]), [], data.num_neuron(a));
%     v = zscore_nan(v, [], 1);
%     data.S_trial{a} = permute(reshape(v, data.trial_length, data.num_trial, data.num_neuron(a)), [3,2,1]);
% end


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
    if any(cellfun(@(x) length(x), trial_idx)<=5)
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
    if exist(result_file); continue; end

    
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
                % v = zscore_nan(v, [], 1);
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
    
    %% compute psth
    % S_sess_resid = S_sess;
    % for m = 1:num_sess
    %     trial_label = data.trial_vec(sess_idx{m});
    %     trial_label(trial_label>=4) = 0;
    %     for a = 1:2
    %         for n = 1:2
    %             for tt = 1:tnum
    %                 idx = trial_label==tt;
    %                 s_avg = nanmean(S_sess{m,a,n}(:,idx,:),2);
    %                 S_sess_resid{m,a,n}(:,idx,:) = S_sess{m,a,n}(:,idx,:) - repmat(s_avg,1,sum(idx),1);
    %             end
    %         end
    %     end
    % end

    S_sess_resid = S_sess;

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

    %% cca section
    cca_tw = {ts{2}, ts{3}, ts{4}, ts{5}};
    ntw = length(cca_tw);
    
    %% cross area full model by subsets by trial type
    ncv_tt_sub = nan(num_sess, ntw, tnum, 2);
    cca_model_tt_sub = cell(num_sess, ntw, tnum, 2, 2);
    cca_model_tt_sub_shuff = cell(num_sess, ntw, tnum, 2, 2, num_shuff);
    cca_r_tt_sub = zeros(num_sess, ntw, tnum, 2);
    cca_r_tt_sub_shuff = zeros(num_sess, ntw, tnum, 2, num_shuff);

    for m = 1:num_sess
        for n = 1:ntw
            for tt = 1:tnum
                for i = 1:2
                    t = cca_tw{n}; 
                    t = t(t>=1 & t<=data.trial_length);
                    nx = S_num(m,1,i); ny = S_num(m,2,i); nxy = min(nx, ny);
                    X = reshape(permute(S_sess_resid{m,1,i}(:,t_idx_sess_idx{m}==tt,t), ...
                        [3,2,1]), [], nx);
                    Y = reshape(permute(S_sess_resid{m,2,i}(:,t_idx_sess_idx{m}==tt,t), ...
                        [3,2,1]), [], ny);
                    keep_idx = (~isnan(X(:,1))) & (~isnan(Y(:,1)));
                    X = X(keep_idx,:); Y = Y(keep_idx,:);
                    nt = sum(keep_idx);
                    
                    if ~isempty(X) && ~isempty(Y)
%                     try
                        [Ai, Bi, r] = canoncorr(X,Y);
                        cca_model_tt_sub{m,n,tt,1,i} = padarray(Ai,[0,nxy-size(Ai,2)], NaN, 'post');
                        cca_model_tt_sub{m,n,tt,2,i} = padarray(Bi,[0,nxy-size(Bi,2)], NaN, 'post');
                        cca_r_tt_sub(m,n,tt,i) = r(1);

                        % shuffled
                        rs = nan(1,num_shuff);
                        Ai = nan(nx,nxy,num_shuff); Bi = nan(ny,nxy,num_shuff); 
                        Y0 = S_sess_resid{m,2,i}(:,t_idx_sess_idx{m}==tt,t);
                        N = sum(t_idx_sess_idx{m}==tt);
%                         for s = 1:num_shuff
                        parfor s = 1:num_shuff
                            s_idx = randperm(nt);
                            if strcmp(shuffle_method, 'trial')
                                s_idx = randperm(N);
                                Xs = X; Ys = Y0; 
                                Ys = reshape(permute(Ys(:,s_idx,:),[3,2,1]), [], ny);
                                Ys = Ys(keep_idx,:);
                                keep_idx2 = (~isnan(Xs(:,1))) & (~isnan(Ys(:,1)));
                                Xs = Xs(keep_idx2,:); Ys = Ys(keep_idx2,:);
                            elseif strcmp(shuffle_method, 'frame')
                                Xs = X; Ys = Y; 
                                Ys = Ys(randperm(nt),:);
                            elseif strcmp(shuffle_method, 'all')
                                Xs = X; Ys = Y(:);
                                Ys = reshape(Ys(randperm(length(Ys))), nt, ny);
                            end
                            if size(Xs,1)>2
                                [Ais, Bis, rsi] = canoncorr(Xs,Ys);
                                Ais = padarray(Ais, [0,nxy-size(Ais,2)], NaN, 'post');
                                Bis = padarray(Bis, [0,nxy-size(Bis,2)], NaN, 'post');
                                Ai(:,:,s) = Ais;  Bi(:,:,s) = Bis;
                                rs(s) = rsi(1);
                            end
                        end
                        cca_r_tt_sub_shuff(m,n,tt,i,:) = rs;
                        cca_model_tt_sub_shuff{m,n,tt,1,i} = Ai;
                        cca_model_tt_sub_shuff{m,n,tt,2,i} = Bi;
                    
                        ncv_tt_sub(m,n,tt,i) = sum(r > nanmean(rs)+sig_dim_thr*nanstd(rs));
                    else
%                     catch ME
                        cca_model_tt_sub{m,n,tt,1,i} = nan(nx ,nxy);
                        cca_model_tt_sub{m,n,tt,2,i} = nan(ny, nxy);
                        cca_model_tt_sub_shuff{m,n,tt,1,i} = nan(nx,nxy,num_shuff);
                        cca_model_tt_sub_shuff{m,n,tt,2,i} = nan(ny,nxy,num_shuff); 
                    end
                end
            end
        end
    end

    %% within area full model by subsets
    ncv_within_tt = nan(num_sess, ntw, tnum, 2);
    cca_model_within_tt = cell(num_sess, ntw, tnum, 2, 2);
    cca_model_within_tt_shuff = cell(num_sess, ntw, tnum, 2, 2, num_shuff);
    cca_r_within_tt = zeros(num_sess, ntw, tnum, 2);
    cca_r_within_tt_shuff = zeros(num_sess, ntw, tnum, 2, num_shuff);

    for m = 1:num_sess
        for n = 1:ntw
            for tt = 1:tnum
                for a = 1:2
                    t = cca_tw{n}; 
                    t = t(t>=1 & t<=data.trial_length);
                    nx = S_num(m,a,1); ny = S_num(m,a,2); nxy = min(nx, ny);
                    X = reshape(permute(S_sess_resid{m,a,1}(:,t_idx_sess_idx{m}==tt,t), ...
                        [3,2,1]), [], nx);
                    Y = reshape(permute(S_sess_resid{m,a,2}(:,t_idx_sess_idx{m}==tt,t), ...
                        [3,2,1]), [], ny);
%                     X = nanmean(S_sess_resid{m,a,1}(:,t_idx_sess{m,tt},t), 3)';
%                     Y = nanmean(S_sess_resid{m,a,2}(:,t_idx_sess{m,tt},t), 3)';
                    keep_idx = (~isnan(X(:,1))) & (~isnan(Y(:,1)));
                    X = X(keep_idx,:); Y = Y(keep_idx,:);
                    nt = sum(keep_idx);

                    if ~isempty(X) && ~isempty(Y)
%                     try
                        [Ai, Bi, r] = canoncorr(X,Y);
                        cca_model_within_tt{m,n,tt,a,1} = padarray(Ai,[0,nxy-size(Ai,2)], NaN, 'post');
                        cca_model_within_tt{m,n,tt,a,2} = padarray(Bi,[0,nxy-size(Bi,2)], NaN, 'post');
                        cca_r_within_tt(m,n,tt,a) = r(1);

                        % shuffled
                        rs = nan(1,num_shuff);
                        Ai = nan(nx,nxy,num_shuff); Bi = nan(ny,nxy,num_shuff); 
                        Y0 = S_sess_resid{m,a,2}(:,t_idx_sess_idx{m}==tt,t);
                        N = sum(t_idx_sess_idx{m}==tt);
                        parfor s = 1:num_shuff
                            s_idx = randperm(nt);
                            if strcmp(shuffle_method, 'trial')
                                s_idx = randperm(N);
                                Xs = X; Ys = Y0; 
                                Ys = reshape(permute(Ys(:,s_idx,:),[3,2,1]), [], ny);
                                Ys = Ys(keep_idx,:);
                                keep_idx2 = (~isnan(Xs(:,1))) & (~isnan(Ys(:,1)));
                                Xs = Xs(keep_idx2,:); Ys = Ys(keep_idx2,:);
                            elseif strcmp(shuffle_method, 'frame')
                                Xs = X; Ys = Y; 
                                Ys = Ys(randperm(nt),:);
                            elseif strcmp(shuffle_method, 'all')
                                Xs = X; Ys = Y(:);
                                Ys = reshape(Ys(randperm(length(Ys))), nt, ny);
                            end
                            if size(Xs,1)>2
                                [Ais, Bis, rsi] = canoncorr(Xs,Ys);
                                Ais = padarray(Ais, [0,nxy-size(Ais,2)], NaN, 'post');
                                Bis = padarray(Bis, [0,nxy-size(Bis,2)], NaN, 'post');
                                Ai(:,:,s) = Ais;  Bi(:,:,s) = Bis;
                                rs(s) = rsi(1);
                            end
                        end
                        cca_r_within_tt_shuff(m,n,tt,a,:) = rs;
                        cca_model_within_tt_shuff{m,n,tt,a,1} = Ai;
                        cca_model_within_tt_shuff{m,n,tt,a,2} = Bi;
                        
                        ncv_within_tt(m,n,tt,a) = sum(r > nanmean(rs)+sig_dim_thr*nanstd(rs));
                    else
%                     catch ME
                        cca_model_within_tt{m,n,tt,a,1} = nan(nx, nxy);
                        cca_model_within_tt{m,n,tt,a,2} = nan(ny, nxy);
                        cca_model_within_tt_shuff{m,n,tt,a,1} = nan(nx,nxy,num_shuff);
                        cca_model_within_tt_shuff{m,n,tt,a,2} = nan(ny,nxy,num_shuff); 
                    end
                end
            end
        end
    end

    
    %% svm vs cca inter axis
    svm_cca_sim_tt = nan(num_sess, K, ntw, tnum, 2, 2);
    svm_cca_sim_tt_shuff = nan(num_sess, K, ntw, tnum, 2, num_shuff, 2);
    for m = 1:num_sess
        for a = 1:2
            for sub_idx = 1:2
                for tt = 1:tnum
                    b1 = cell2mat(task_axis(m,:,1,a,sub_idx))';  % correct trial axis
                    b2 = cellfun(@(x) x(:,1), cca_model_tt_sub(m,:,tt,a,sub_idx), 'uniformoutput', 0);
                    b2 = cell2mat(b2)';
                    if weighted_sim  % weight similarity
                        w = pca_weight{m,a,sub_idx};
                        dist_fun = @(u,v) (w.*u*v') ./ (sqrt(sum(w.*(u.^2))) * sqrt(w*(v.^2)'));
                        svm_cca_sim_tt(m,:,:,tt,a,sub_idx) = pdist2(b1, b2, dist_fun);
                    else
                        svm_cca_sim_tt(m,:,:,tt,a,sub_idx) = 1-pdist2(b1, b2, 'cos');
                    end
                    for s = 1:num_shuff
                        if strcmp(sim_shuff_method, 'model')
                            b1s = b1;
                            b2s = cellfun(@(x) x(:,1,s), cca_model_tt_sub_shuff...
                                (m,:,tt,a,sub_idx), 'uniformoutput', 0);
                            b2s = cell2mat(b2s)';
                        elseif strcmp(sim_shuff_method, 'perm')
                            b1s = b1(:,randperm(S_num(m,a,sub_idx)));
                            b2s = b2(:,randperm(S_num(m,a,sub_idx)));
                        elseif strcmp(sim_shuff_method, 'const')
                            b1s = b1; b2s = ones(size(b1));
                        end
                        if weighted_sim  % weight similarity
                            svm_cca_sim_tt_shuff(m,:,:,tt,a,s,sub_idx) = pdist2(b1s, b2s, dist_fun);
                        else
                            svm_cca_sim_tt_shuff(m,:,:,tt,a,s,sub_idx) = 1-pdist2(b1s, b2s, 'cos');
                        end
                    end
                end
            end
        end
    end

    %% svm vs cca within axis
    svm_cca_sim_tt_within = nan(num_sess, K, ntw, tnum, 2, 2);
    svm_cca_sim_tt_within_shuff = nan(num_sess, K, ntw, tnum, 2, num_shuff, 2);
    for m = 1:num_sess
        for a = 1:2
            for sub_idx = 1:2
                for tt = 1:tnum
                    b1 = cell2mat(task_axis(m,:,1,a,sub_idx))';  % correct trial axis
                    b2 = cellfun(@(x) x(:,1), cca_model_within_tt(m,:,tt,a,sub_idx), 'uniformoutput', 0);
                    b2 = cell2mat(b2)';
                    if weighted_sim  % weight similarity
                        w = pca_weight{m,a,sub_idx};
                        dist_fun = @(u,v) (w.*u*v') ./ (sqrt(sum(w.*(u.^2))) * sqrt(w*(v.^2)'));
                        svm_cca_sim_tt_within(m,:,:,tt,a,sub_idx) = pdist2(b1, b2, dist_fun);
                    else
                        svm_cca_sim_tt_within(m,:,:,tt,a,sub_idx) = 1-pdist2(b1,b2,'cos');
                    end
                    for s = 1:num_shuff
                        if strcmp(sim_shuff_method, 'model')
                            b1s = b1;
                            b2s = cellfun(@(x) x(:,1,s), cca_model_within_tt_shuff...
                                (m,:,tt,a,sub_idx), 'uniformoutput', 0);
                            b2s = cell2mat(b2s)';
                        elseif strcmp(sim_shuff_method, 'perm')
                            b1s = b1(:,randperm(S_num(m,a,sub_idx)));
                            b2s = b2(:,randperm(S_num(m,a,sub_idx)));
                        elseif strcmp(sim_shuff_method, 'const')
                            b1s = b1; b2s = ones(size(b1));
                        end
                        if weighted_sim  % weight similarity
                            svm_cca_sim_tt_within_shuff(m,:,:,tt,a,s,sub_idx) = pdist2(b1s,b2s,dist_fun);
                        else
                            svm_cca_sim_tt_within_shuff(m,:,:,tt,a,s,sub_idx) = 1-pdist2(b1s,b2s,'cos');
                        end
                    end
                end
            end
        end
    end


    %% cca inter similarity sub models
    cca_sim_tt_sub = nan(num_sess, ntw, ntw, tnum, 2, 2);
    cca_sim_tt_sub_shuff = nan(num_sess, ntw, ntw, tnum, 2, num_shuff, 2);
    for m = 1:num_sess
        for a = 1:2
            for sub_idx = 1:2
                for tt = 1:tnum
                    b = cellfun(@(x) x(:,1), cca_model_tt_sub(m,:,tt,a,sub_idx), 'uniformoutput', 0);
                    b = cell2mat(b)';
                    if weighted_sim  % weight similarity
                        w = pca_weight{m,a,sub_idx};
                        dist_fun = @(u,v) (w.*u*v') ./ (sqrt(sum(w.*(u.^2))) * sqrt(w*(v.^2)'));
                        cca_sim_tt_sub(m,:,:,tt,a,sub_idx) = pdist2(b, b, dist_fun);
                    else
                        cca_sim_tt_sub(m,:,:,tt,a,sub_idx) = 1-pdist2(b, b, 'cos');
                    end
                    for s = 1:num_shuff
                        if strcmp(sim_shuff_method, 'model')
                            bs = cellfun(@(x) x(:,1,s), cca_model_tt_sub_shuff...
                                (m,:,tt,a,sub_idx), 'uniformoutput', 0);
                            bs = cell2mat(bs)';
                            b1s = bs; b2s = bs;
                        elseif strcmp(sim_shuff_method, 'perm')
                            bs = b(:,randperm(S_num(m,a,sub_idx)));
                            b1s = bs; b2s = bs;
                        elseif strcmp(sim_shuff_method, 'const')
                            b1s = b; b2s = ones(size(b));
                        end
                        if weighted_sim  % weight similarity
                            cca_sim_tt_sub_shuff(m,:,:,tt,a,s,sub_idx) = pdist2(b1s,b2s,dist_fun);
                        else
                            cca_sim_tt_sub_shuff(m,:,:,tt,a,s,sub_idx) = 1-pdist2(b1s,b2s,'cos');
                        end
                    end
                end
            end
        end
    end


    %% cca within similarity
    cca_sim_within_tt = nan(num_sess, ntw, ntw, tnum, 2, 2);
    cca_sim_within_tt_shuff = nan(num_sess, ntw, ntw, tnum, 2, num_shuff, 2);
    for m = 1:num_sess
        for a = 1:2
            for sub_idx = 1:2
                for tt = 1:tnum
                    b = cellfun(@(x) x(:,1), cca_model_within_tt(m,:,tt,a,sub_idx), 'uniformoutput', 0);
                    b = cell2mat(b)';
                    if weighted_sim  % weight similarity
                        w = pca_weight{m,a,sub_idx};
                        dist_fun = @(u,v) (w.*u*v') ./ (sqrt(sum(w.*(u.^2))) * sqrt(w*(v.^2)'));
                        cca_sim_within_tt(m,:,:,tt,a,sub_idx) = pdist2(b, b, dist_fun);
                    else
                        cca_sim_within_tt(m,:,:,tt,a,sub_idx) = 1-pdist2(b, b, 'cos');
                    end
                    for s = 1:num_shuff
                        if strcmp(sim_shuff_method, 'model')
                            bs = cellfun(@(x) x(:,1,s), cca_model_within_tt_shuff...
                                (m,:,tt,a,sub_idx), 'uniformoutput', 0);
                            bs = cell2mat(bs)';
                            b1s = bs; b2s = bs;
                        elseif strcmp(sim_shuff_method, 'perm')
                            bs = b(:,randperm(S_num(m,a,sub_idx)));
                            b1s = bs; b2s = bs;
                        elseif strcmp(sim_shuff_method, 'const')
                            b1s = b; b2s = ones(size(b));
                        end
                        if weighted_sim  % weight similarity
                            cca_sim_within_tt_shuff(m,:,:,tt,a,s,sub_idx) = pdist2(b1s ,b2s, dist_fun);
                        else
                            cca_sim_within_tt_shuff(m,:,:,tt,a,s,sub_idx) = 1-pdist2(b1s,b2s,'cos');
                        end
                    end
                end
            end
        end
    end

    %% cca inter vs within similarity
    cca_sim_inter_vs_within_tt = nan(num_sess, ntw, ntw, tnum, 2, 2);
    cca_sim_inter_vs_within_tt_shuff = nan(num_sess, ntw, ntw, tnum, 2, 2);
    for m = 1:num_sess
        for a = 1:2
            for sub_idx = 1:2
                for tt = 1:tnum
                    b1 = cellfun(@(x) x(:,1), cca_model_tt_sub(m,:,tt,a,sub_idx), 'uniformoutput', 0);
                    b1 = cell2mat(b1)';
                    b2 = cellfun(@(x) x(:,1), cca_model_within_tt(m,:,tt,a,sub_idx), 'uniformoutput', 0);
                    b2 = cell2mat(b2)';
                    if weighted_sim  % weight similarity
                        w = pca_weight{m,a,sub_idx};
                        dist_fun = @(u,v) (w.*u*v') ./ (sqrt(sum(w.*(u.^2))) * sqrt(w*(v.^2)'));
                        cca_sim_inter_vs_within_tt(m,:,:,tt,a,sub_idx) = pdist2(b1, b2, dist_fun);
                    else
                        cca_sim_inter_vs_within_tt(m,:,:,tt,a,sub_idx) = 1-pdist2(b1, b2, 'cos');
                    end
                    for s = 1:num_shuff
                        if strcmp(sim_shuff_method, 'model')
                            b1s = cellfun(@(x) x(:,1,s), cca_model_tt_sub_shuff...
                                (m,:,tt,a,sub_idx), 'uniformoutput', 0);
                            b1s = cell2mat(b1s)';
                            b2s = cellfun(@(x) x(:,1,s), cca_model_within_tt_shuff...
                                (m,:,tt,a,sub_idx), 'uniformoutput', 0);
                            b2s = cell2mat(b2s)';
                        elseif strcmp(sim_shuff_method, 'perm')
                            b1s = b1(:,randperm(S_num(m,a,sub_idx)));
                            b2s = b2(:,randperm(S_num(m,a,sub_idx)));
                        elseif strcmp(sim_shuff_method, 'const')
                            b1s = b; b2s = ones(size(b));
                        end
                        if weighted_sim  % weight similarity
                            cca_sim_inter_vs_within_tt_shuff(m,:,:,tt,a,s,sub_idx) = pdist2(b1s,b2s,dist_fun);
                        else
                            cca_sim_inter_vs_within_tt_shuff(m,:,:,tt,a,s,sub_idx) = 1-pdist2(b1s,b2s,'cos');
                        end
                    end
                end
            end
        end
    end
    
    %% cca inter similarity sub models, correct vs incorrect
    cca_sim_tt_between = nan(num_sess, ntw, ntw, 2, 2);
    cca_sim_tt_between_shuff = nan(num_sess, ntw, ntw, 2, num_shuff, 2);
    for m = 1:num_sess
        for a = 1:2
            for sub_idx = 1:2
                b1 = cellfun(@(x) x(:,1), cca_model_tt_sub(m,:,1,a,sub_idx), 'uniformoutput', 0);
                b1 = cell2mat(b1)';
                b2 = cellfun(@(x) x(:,1), cca_model_tt_sub(m,:,2,a,sub_idx), 'uniformoutput', 0);
                b2 = cell2mat(b2)';
                if weighted_sim  % weight similarity
                    w = pca_weight{m,a,sub_idx};
                    dist_fun = @(u,v) (w.*u*v') ./ (sqrt(sum(w.*(u.^2))) * sqrt(w*(v.^2)'));
                    cca_sim_tt_between(m,:,:,a,sub_idx) = pdist2(b1, b2, dist_fun);
                else
                    cca_sim_tt_between(m,:,:,a,sub_idx) = 1-pdist2(b1, b2, 'cos');
                end
                for s = 1:num_shuff
                    if strcmp(sim_shuff_method, 'model')
                        b1s = cellfun(@(x) x(:,1,s), cca_model_tt_sub_shuff...
                            (m,:,1,a,sub_idx), 'uniformoutput', 0);
                        b1s = cell2mat(b1s)';
                        b2s = cellfun(@(x) x(:,2,s), cca_model_tt_sub_shuff...
                            (m,:,2,a,sub_idx), 'uniformoutput', 0);
                        b2s = cell2mat(b2s)';
                    elseif strcmp(sim_shuff_method, 'perm')
                        b1s = b1(:,randperm(S_num(m,a,sub_idx)));
                        b2s = b2(:,randperm(S_num(m,a,sub_idx)));
                    elseif strcmp(sim_shuff_method, 'const')
                        b1s = b1; b2s = ones(size(b));
                    end
                    if weighted_sim  % weight similarity
                        cca_sim_tt_between_shuff(m,:,:,a,s,sub_idx) = pdist2(b1s,b2s,dist_fun);
                    else
                        cca_sim_tt_between_shuff(m,:,:,a,s,sub_idx) = 1-pdist2(b1s,b2s,'cos');
                    end
                end
            end
        end
    end
    
    %% cca within similarity, correct vs incorrect
    cca_sim_within_tt_between = nan(num_sess, ntw, ntw, 2);
    cca_sim_within_tt_between_shuff = nan(num_sess, ntw, ntw, 2, num_shuff, 2);
    for m = 1:num_sess
        for a = 1:2
            for sub_idx = 1:2
                b1 = cellfun(@(x) x(:,1), cca_model_within_tt(m,:,1,a,sub_idx), 'uniformoutput', 0);
                b1 = cell2mat(b1)';
                b2 = cellfun(@(x) x(:,1), cca_model_within_tt(m,:,2,a,sub_idx), 'uniformoutput', 0);
                b2 = cell2mat(b2)';
                if weighted_sim  % weight similarity
                    w = pca_weight{m,a,sub_idx};
                    dist_fun = @(u,v) (w.*u*v') ./ (sqrt(sum(w.*(u.^2))) * sqrt(w*(v.^2)'));
                    cca_sim_within_tt_between(m,:,:,a,sub_idx) = pdist2(b1, b2, dist_fun);
                else
                    cca_sim_within_tt_between(m,:,:,a,sub_idx) = 1-pdist2(b1, b2, 'cos');
                end
                for s = 1:num_shuff
                    if strcmp(sim_shuff_method, 'model')
                        b1s = cellfun(@(x) x(:,1,s), cca_model_within_tt_shuff...
                            (m,:,1,a,sub_idx), 'uniformoutput', 0);
                        b1s = cell2mat(b1s)';
                        b2s = cellfun(@(x) x(:,1,s), cca_model_within_tt_shuff...
                            (m,:,2,a,sub_idx), 'uniformoutput', 0);
                        b2s = cell2mat(b2s)';
                    elseif strcmp(sim_shuff_method, 'perm')
                        b1s = b1(:,randperm(S_num(m,a,sub_idx)));
                        b2s = b2(:,randperm(S_num(m,a,sub_idx)));
                    elseif strcmp(sim_shuff_method, 'const')
                        b1s = b1; b2s = ones(size(b2));
                    end
                    if weighted_sim  % weight similarity
                        cca_sim_within_tt_between_shuff(m,:,:,a,s,sub_idx) = pdist2(b1s ,b2s, dist_fun);
                    else
                        cca_sim_within_tt_between_shuff(m,:,:,a,s,sub_idx) = 1-pdist2(b1s,b2s,'cos');
                    end
                end
            end
        end
    end
    
    %% cca axis projection task AUC
    cca_within_auc_tt = nan(num_sess, K, ntw, tnum, 2, 2);
    cca_within_auc_tt_shuff = nan(num_sess, K, ntw, tnum, 2, num_shuff2, 2);
    cca_inter_auc_tt = nan(num_sess, K, ntw, tnum, 2, 2);
    cca_inter_auc_tt_shuff = nan(num_sess, K, ntw, tnum, 2, num_shuff2, 2);
    
    for k = 1:K
        for m = 1:num_sess
            for tt = 1:tnum
                Y = label_sess{m,k}(t_idx_sess_idx{m}==tt)';
                keep_idx = (~isnan(Y)) & (Y~=0);
                Y = Y(keep_idx);
                nt = sum(keep_idx);
                if isempty(Y); continue; end
                for a = 1:2
                    for sub_idx = 1:2
                        X = S_sess{m,a,sub_idx}(:,t_idx_sess_idx{m}==tt,:);
                        X = X(:,keep_idx,:); 
                        N = S_num(m,a,sub_idx);
                        for t = 1:ntw
                            Xn = nanmean(X(:,:,cca_tw{t}), 3)';

                            % within area axis projection
                            w = cca_model_within_tt{m,t,tt,a,sub_idx}(:,1);
                            x = Xn*w;
                            try; [~,~,~,cca_within_auc_tt(m,k,t,tt,a,sub_idx)] = perfcurve(Y, x, 2);
                            catch ME; continue; 
                            end
                            % shuffled models
    %                         xs = Xn*permute(cca_model_within_shuff{m,t,a,sub_idx}(:,1,:), [1,3,2]);
                            p = nan(num_shuff2,1);
                            for s = 1:num_shuff2
                                xs = Xn;
                                xs = xs(randperm(nt),randperm(N))*w;
                                try; [~,~,~,p(s)] = perfcurve(Y, xs, 2);
    %                             try; [~,~,~,p(s)] = perfcurve(Y, xs(:,s), 2);
                                catch ME; continue; 
                                end
                            end
                            cca_within_auc_tt_shuff(m,k,t,tt,a,:,sub_idx) = p;
                            
                            % inter area axis projection
                            w = cca_model_tt_sub{m,t,tt,a,sub_idx}(:,1);
                            x = Xn*w;
                            try; [~,~,~,cca_inter_auc_tt(m,k,t,tt,a,sub_idx)] = perfcurve(Y, x, 2);
                            catch ME; continue; 
                            end
                            % shuffled models
    %                         xs = Xn*permute(cca_model_within_shuff{m,t,a,sub_idx}(:,1,:), [1,3,2]);
                            p = nan(num_shuff2,1);
                            for s = 1:num_shuff2
                                xs = Xn;
                                xs = xs(randperm(nt),randperm(N))*w;
                                try; [~,~,~,p(s)] = perfcurve(Y, xs, 2);
    %                             try; [~,~,~,p(s)] = perfcurve(Y, xs(:,s), 2);
                                catch ME; continue; 
                                end
                            end
                            cca_inter_auc_tt_shuff(m,k,t,tt,a,:,sub_idx) = p;
                        end
                    end
                end
            end
        end
    end

%% 
if plot_result

%% prediction auc
m = 1;
tw = [2,3,4,5];
cc = turbo(5);
figure; set(gcf,'color','w');
for a = 1:2
    subplot(1, 2, a); hold on; 
    for sub_idx = 1:2
        v = pred_auc(m,:,a,sub_idx);
        v = abs(v-0.5)*2;
        plot(v, 'color', cc(sub_idx,:));
    end
    set(gca, 'xtick', 1:K, 'xticklabel', label_str, 'xticklabelrotation', 45);
    ylabel(sprintf('%s\nAUC', area_str{a})); box off;
end

%% similarity
figure; set(gcf, 'color', 'w');
for k = 1:K
    for a = 1:2
        subplot(2,K,(a-1)*K+k);
        v = abs(squeeze(svm_cca_sim(m,:,:,k,a)));
        imagesc(v);
        caxis([0 0.5])
        if a==1; title(label_str{k}); end
    end
end

end
    
%% save
save(result_file, 'beh_rate', 'sess_idx', 'num_sess', 'S_num',...
    'var_explained', 'task_axis', 'ts', 't_idx_sess',...
    'pred_auc', 'pred_auc_shuff', ...
    'ncv_tt_sub', 'cca_r_tt_sub', 'cca_r_tt_sub_shuff', 'ncv_within_tt', ...
    'cca_r_within_tt', 'cca_r_within_tt_shuff', ...
    'svm_cca_sim_tt', 'svm_cca_sim_tt_within', 'cca_sim_tt_sub', ...
    'cca_sim_within_tt', 'cca_sim_inter_vs_within_tt', 'svm_cca_sim_tt_shuff', ...
    'svm_cca_sim_tt_within_shuff', 'cca_sim_tt_sub_shuff', 'cca_sim_within_tt_shuff',...
    'cca_sim_inter_vs_within_tt_shuff', ...
    'cca_within_auc_tt', 'cca_within_auc_tt_shuff', ...
    'cca_inter_auc_tt', 'cca_inter_auc_tt_shuff', ...
    'cca_sim_tt_between', 'cca_sim_tt_between_shuff', ...
    'cca_sim_within_tt_between', 'cca_sim_within_tt_between_shuff',...
    '-v7.3');
toc;

end
    
end



