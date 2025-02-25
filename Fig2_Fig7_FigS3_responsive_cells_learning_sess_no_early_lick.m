addpath(genpath('./'));

addpath('../tensor_toolbox_2.6');
addpath('../tensor_toolbox_2.6/met');
addpath('../tensor-demo-master/matlab');
addpath('../nonnegfac-matlab-master');

%%
clear; clc;
warning off;
setup_colors;
datasheet = get_data_sheet('multiarea');

opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p';
result_name = 'responsive_neuron_learning_no_early_lick_sess_120';
 
dataset = [131:186,188:256, 389:672];
var_to_read = {'S_trial', 'num_neuron', 'num_trial', 'trial_vec',...
    'trial_length', 'ts', 'ts_fr', 'first_correct_lick', 'choice_time',...
    'task_label', 'S_rew', 'F0'};
plot_result = 0;

sess_len = 120; min_sess_len = 5;

standard_trial_structure = [0.5,1,1,0.5,2,4];
standard_fr = 10;
standard_ts = make_trial_structure(standard_trial_structure, standard_fr);
standard_ts_fr = cellfun(@(x) x/standard_fr, standard_ts, 'uniformoutput', false);
trial_len = standard_ts{end}(end);
tvec = (1:trial_len)/standard_fr;

%%
for dataid = dataset

dinfo = data_info(datasheet, dataid, 'multiarea', opts.base_dir);
if dinfo.quality_idx==0; continue; end
spath = fullfile(dinfo.work_dir, opts.result_dir);

exp_id = get_exp_condition_idx(dinfo);

fprintf('dataset %d...\n',dataid);
data = load_data(dinfo, var_to_read, opts);

if isempty(data.ts{4}); data.ts{4} = data.ts{3}(end):data.ts{5}(1); end
if isempty(data.ts{1}); data.ts{1} = 1; end
if data.ts{2}(1)==0; data.ts{2} = data.ts{2}(2:end); end

fr = dinfo.fr;
num_ts = length(data.ts);
num_trial_type = 14;
tnum = num_trial_type/2;

%% normalize data
f = fspecial('gaussian',[1,3], 1);
for a = 1:2
    
    data.F0{a}(data.F0{a}==0) = 1;
    data.S_trial{a} = data.S_trial{a}./repmat(data.F0{a}, 1, data.num_trial, data.trial_length); 
    
    % smooth with a gaussian kernal
    data.S_trial{a} = smooth_spike_data(data.S_trial{a}, f);
    
    % zscore
    v = reshape(permute(data.S_trial{a}, [3,2,1]), [], data.num_neuron(a));
    v = zscore_nan(v,[],1);
    data.S_trial{a} = permute(reshape(v, data.trial_length, data.num_trial,...
        data.num_neuron(a)), [3,2,1]);
    
    % align data
    data.S_rew{a} = align_data_by_choice(data.S_trial{a}, data.choice_time, data.ts);
    data.S_rew_shift{a} = align_data_by_choice_shift(data.S_trial{a}, data.choice_time, data.ts);
end

%% handle choice window
choice_tw = 5;
S_data = cell(1,2);
S_ts = data.ts;
for a = 1:2
    S_data{a} = data.S_rew{a};
    S_data{a}(:,:,data.ts{3}(end)+1:data.ts{3}(end)+choice_tw) = ...
        data.S_rew_shift{a}(:,:,data.ts{4}(end)-choice_tw+1:data.ts{4}(end));
    S_data{a} = S_data{a}(:,:,[1:data.ts{3}(end)+choice_tw, data.ts{5}(1):end]);
end
S_ts{4} = S_ts{3}(end)+1:S_ts{3}(end)+choice_tw;
S_ts{5} = (S_ts{4}(end)+1) : (S_ts{4}(end)+1) + length(S_ts{5})-1;
S_ts{6} = (S_ts{5}(end)+1) : size(S_data{1},3);
data.trial_length = S_ts{6}(end);

data.task_label.err_vec = data.task_label.rew_vec;
data.task_label.err_vec(data.task_label.err_vec>0) = 2;
data.task_label.err_vec(data.task_label.err_vec==0) = 1;

%% remove early licks 
for a = 1:2
    for i = 1:data.num_trial
        lick_frame = data.first_correct_lick(i);
        if isnan(lick_frame); continue; end
        idx_remove = lick_frame:S_ts{3}(end);
        S_data{a}(:,i,idx_remove) = NaN;
    end
end


%% align data to standard
for a = 1:2
    S_data{a} = align_trial_to_standard(S_data{a}, S_ts, standard_ts);
end

data.trial_length = trial_len;
S_ts = standard_ts;


%% remove mismatch trials
keep_idx = data.trial_vec<=4;
for a = 1:2
    S_data{a} = S_data{a}(:,keep_idx,:);
end
data.trial_vec = data.trial_vec(keep_idx);
data.num_trial = sum(keep_idx);
data.task_label.cue_vec = data.task_label.cue_vec(keep_idx);
data.task_label.tex_vec = data.task_label.tex_vec(keep_idx);
data.task_label.choice_vec = data.task_label.choice_vec(keep_idx);
data.task_label.rew_vec = data.task_label.rew_vec(keep_idx);
data.task_label.err_vec = data.task_label.err_vec(keep_idx);

%% split sessions
sess_idx = split_session(data.num_trial, sess_len, min_sess_len);
num_sess = length(sess_idx);

% behavior rate for each session
beh_rate = zeros(1,num_sess);
miss_rate = zeros(1,num_sess);
for n = 1:num_sess
    n_act = sum(data.trial_vec(sess_idx{n})==1) + sum(data.trial_vec(sess_idx{n})==2) ...
        + sum(data.trial_vec(sess_idx{n})==3) + sum(data.trial_vec(sess_idx{n})==4);
    n_correct = sum(data.trial_vec(sess_idx{n})==1) + sum(data.trial_vec(sess_idx{n})==2);
    beh_rate(n) = n_correct/n_act*100;
end

%% trial labels
trial_label = {data.task_label.cue_vec, data.task_label.tex_vec, ...
    data.task_label.choice_vec, data.task_label.rew_vec, ...
    data.task_label.err_vec};
tw_label = [2,3,4,5,5];
label_str = {'Tone', 'Texture', 'Choice', 'Reward', 'Error'};
ntw = length(trial_label);
ts = {S_ts{2}, S_ts{3}, S_ts{4}, S_ts{5}, S_ts{5}};

%% responsive neurons
p_thr = 0.05; 
num_shuff = 100;
responsive_neuron = cell(num_sess,2,ntw);
for m = 1:num_sess
    for a = 1:2
        for i = 1:data.num_neuron(a)
            f = squeeze(S_data{a}(i,sess_idx{m},:));
            for n = 1:ntw
                v = f(:,ts{n});
                ts_ctrl = setdiff(1:data.trial_length, ts{n});
                v0 = [];
                n1 = length(ts_ctrl); n2 = min(n1, length(ts{n}));
                for s = 1:num_shuff
                    idx = randperm(n1, n2);
                    v0(:,:,s) = f(:,ts_ctrl(idx));
                end
                v = nanmean(v,2); v0 = nanmean(v0,2);

                try; pval = ranksum(v(:), v0(:), 'tail', 'right');
                catch ME; pval = NaN;
                end
                if pval<p_thr && abs(nanmean(v(:)))>abs(nanmean(v0(:)))*1.1
                    responsive_neuron{m,a,n}(end+1) = i;
                end
            end
        end
    end
end

%% task variable tuned neuron
disc_neuron_raw = cell(num_sess,2,ntw);
disc_neuron = cell(num_sess,2,ntw);
trial_group = cell(num_sess,2,ntw);

for m = 1:num_sess
    % group trials
    for n = 1:ntw
        for i = 1:2
            trial_group{m,i,n} = trial_label{n}(sess_idx{m})==i;
            % take only correct trials
            trial_group{m,i,n} = trial_group{m,i,n} & data.trial_vec(sess_idx{m})<=8;
        end
    end

    for n = 1:ntw
        for a = 1:2
            [disc_neuron_raw{m,a,n,2}, disc_neuron_raw{m,a,n,1}] = find_tuned_neuron...
                (S_data{a}(:,sess_idx{m},:), trial_group(m,:,n), ts{n});
            for i = 1:2
                disc_neuron{m,a,n,i} = intersect(responsive_neuron{m,a,n}, disc_neuron_raw{m,a,n,i});
                disc_neuron{m,a,n,i} = reshape(disc_neuron{m,a,n,i}, [], 1);
            end

        end
    end
end

%% overlap
num_responsive_overlap = zeros(num_sess,2,ntw,ntw);
num_disc_overlap = zeros(num_sess,2,ntw,ntw,2);
for m = 1:num_sess
    for a = 1:2
        for n = 1:ntw
            v = responsive_neuron{m,a,n};
            num_responsive_overlap(m,a,n,:) = cellfun(@(x) length(intersect(x,v))/...
                length(unique([x(:);v(:)])),responsive_neuron(m,a,:))*100;
            for i = 1:2
                if i==1
                    v = disc_neuron{m,a,n,1};
                    v1 = cellfun(@(x) length(intersect(x,v))/...
                        length(unique([x(:);v(:)])),disc_neuron(m,a,:,1))*100;
                    v = disc_neuron{m,a,n,2};
                    v2 = cellfun(@(x) length(intersect(x,v))/...
                        length(unique([x(:);v(:)])),disc_neuron(m,a,:,2))*100;
                elseif i==2
                    v = disc_neuron{m,a,n,1};
                    v1 = cellfun(@(x) length(intersect(x,v))/...
                        length(unique([x(:);v(:)])),disc_neuron(m,a,:,2))*100;
                    v = disc_neuron{m,a,n,2};
                    v2 = cellfun(@(x) length(intersect(x,v))/...
                        length(unique([x(:);v(:)])),disc_neuron(m,a,:,1))*100;
                end
                num_disc_overlap(m,a,n,:,i) = nanmean(cat(1,v1,v2),1);
            end
        end
    end
end

%% response avg  *** already aligned
resp_avg = cell(num_sess,2,ntw,tnum);
disc_avg = cell(num_sess,2,ntw,tnum,2);
disc_joint_avg = cell(num_sess,2,ntw,ntw,tnum,2);
for m = 1:num_sess
    trial_vec_sess = data.trial_vec(sess_idx{m});
    for a = 1:2
        S_sess = S_data{a}(:,sess_idx{m},:);
        for t = 1:tnum
            t_idx = trial_vec_sess==2*t-1 | trial_vec_sess==2*t;
            for n = 1:ntw
                v = nanmean(nanmean(S_sess(responsive_neuron{m,a,n},t_idx,:),1),2);
                resp_avg{m,a,n,t} = align_trial_to_standard_2d(permute(v,[2,3,1]), S_ts, standard_ts);
                for i = 1:2  % correct, incorrect
                    if i==1
                        v1 = S_sess(disc_neuron{m,a,n,1},trial_vec_sess==2*t-1,:);
                        v2 = S_sess(disc_neuron{m,a,n,2},trial_vec_sess==2*t,:);
                    else
                        v1 = S_sess(disc_neuron{m,a,n,1},trial_vec_sess==2*t,:);
                        v2 = S_sess(disc_neuron{m,a,n,2},trial_vec_sess==2*t-1,:);
                    end
                    v = cat(2,nanmean(v1,1),nanmean(v2,1));
                    v = nanmean(v,2);
                    disc_avg{m,a,n,t,i} = align_trial_to_standard_2d(permute(v,[2,3,1]), S_ts, standard_ts);
                end

                % joint disc neurons
                for tw = 1:ntw
                    for i = 1:2  % correct, incorrect
                        if i==1
                            n1 = intersect(disc_neuron{m,a,n,1}, disc_neuron{m,a,tw,1});
                            n2 = intersect(disc_neuron{m,a,n,2}, disc_neuron{m,a,tw,2});
                            v1 = S_sess(n1,trial_vec_sess==2*t-1,:);
                            v2 = S_sess(n2,trial_vec_sess==2*t,:);
                        else
                            n1 = intersect(disc_neuron{m,a,n,1}, disc_neuron{m,a,tw,2});
                            n2 = intersect(disc_neuron{m,a,n,2}, disc_neuron{m,a,tw,1});
                            v1 = S_sess(n1,trial_vec_sess==2*t-1,:);
                            v2 = S_sess(n2,trial_vec_sess==2*t,:);
                        end
                        v = cat(2,nanmean(v1,1),nanmean(v2,1));
                        v = nanmean(v,2);
                        disc_joint_avg{m,a,n,tw,t,i} = align_trial_to_standard_2d...
                            (permute(v,[2,3,1]), S_ts, standard_ts);
                    end
                end

            end
        end
    end
end


%% save
save(fullfile(spath, [result_name '.mat']), 'responsive_neuron', ...
    'disc_neuron', 'resp_avg', 'disc_avg', 'num_responsive_overlap', 'num_disc_overlap',...
    'disc_joint_avg', ...
    'sess_idx', 'beh_rate', '-v7.3');

end


