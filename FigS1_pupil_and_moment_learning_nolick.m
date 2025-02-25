

%%
clear all; clc
setup_colors;
datasheet = get_data_sheet('multiarea');

opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p';

var_to_read = {'trial_vec', 'pupil_r', 'body', 'ts', 'trial', 'face',...
    'trial_length', 'rate', 'choice_time', 'first_correct_lick'};

dataset = [132:186, 188:256, 389:672]; 
dataset = setdiff(dataset, 220);

nset = length(dataset);

standard_trial_structure = [0.5,1,1,0.5,2,4];
standard_fr = 10;
standard_ts = make_trial_structure(standard_trial_structure, standard_fr);
standard_ts_fr = cellfun(@(x) x/standard_fr, standard_ts, 'uniformoutput', false);
trial_len = standard_ts{end}(end);
tvec = (1:trial_len)/standard_fr;

tnum = 4;
num_ts = 6;

%% load all data
body =  cell(nset, tnum);
pupil =  cell(nset, tnum);
face =  cell(nset, tnum);

exp_idx = zeros(1,nset); 
beh_rate = nan(1,nset); 
rev_rate = nan(1,nset);
a_idx = false(1,nset); 
mouse_id = zeros(1,nset);
quality_idx = zeros(1,nset);

count = 0;
reverse_str = '';
for dataid = 1:length(dataset)

    dinfo = data_info(datasheet, dataset(dataid), 'multiarea', opts.base_dir);
    % print current progress
    msg = sprintf('Loading dataset %d\n', dataset(dataid));
    fprintf([reverse_str, msg]);
    reverse_str = repmat(sprintf('\b'), 1, length(msg));
    
    spath = fullfile(dinfo.work_dir, opts.result_dir);
    data = load_data(dinfo, var_to_read, opts);
    data.num_trial = length(data.trial_vec);
    fr = dinfo.fr;
    beh_rate(dataid) = data.rate.total_nor;
    
    % for dataset 242, choice window is empty
    if isempty(data.ts{4}); data.ts{4} = data.ts{3}(end):data.ts{5}(1); end
    if isempty(data.ts{1}); data.ts{1} = [1,2]; data.ts{2} = data.ts{2}(data.ts{2}>0); end
    
    if all(isnan(data.pupil_r(:)))
        fprintf('empty pupil file, check dataset %d\n', dataset(dataid));
    end
    
    %% keep only matched trials
    keep_idx = find(data.trial_vec<=4);
    num_trial = length(keep_idx);
    
    data.trial_vec = data.trial_vec(keep_idx);
    data.pupil_r = data.pupil_r(keep_idx,:);
    data.face = data.face(keep_idx,:);
    data.body = data.body(keep_idx,:);
    
    %% split session
    sess_len = 120;  min_sess_len = 5;
    sess_idx = split_session(num_trial, sess_len, min_sess_len);
    num_sess = length(sess_idx);
    
    %% collect data
    exp_idx(count+1:count+num_sess) = get_exp_condition_idx(dinfo);
    if strcmp(dinfo.areas, 'S1/RL')
        a_idx(count+1:count+num_sess) = 0;
    elseif strcmp(dinfo.areas, 'A/RL')
        a_idx(count+1:count+num_sess) = 1; 
    elseif strcmp(dinfo.areas, 'S1/A')
        a_idx(count+1:count+num_sess) = 2;
    end
    mouse_id(count+1:count+num_sess) = dinfo.mouse_id; 
    quality_idx(count+1:count+num_sess) = dinfo.quality_idx;
    
    %% align data
    data.pupil_rew = align_data_by_choice(data.pupil_r, data.choice_time, data.ts);
    data.pupil_rew_shift = align_data_by_choice_shift(data.pupil_rew, data.choice_time, data.ts);
    data.face_rew = align_data_by_choice(data.face, data.choice_time, data.ts);
    data.face_rew_shift = align_data_by_choice_shift(data.face, data.choice_time, data.ts);
    data.body_rew = align_data_by_choice(data.body, data.choice_time, data.ts);
    data.body_rew_shift = align_data_by_choice_shift(data.body, data.choice_time, data.ts);
    
    %% handle choice window
    choice_tw = 5;
    ts = data.ts;
    
    % pupil
    data.pupil_aligned = data.pupil_rew;
    data.pupil_aligned(:,ts{3}(end)+1:ts{3}(end)+choice_tw) = ...
        data.pupil_rew_shift(:,ts{4}(end)-choice_tw+1:ts{4}(end));
    data.pupil_aligned = data.pupil_aligned(:,[1:ts{3}(end)+choice_tw, ts{5}(1):end]);
    % face
    data.face_aligned = data.face_rew;
    data.face_aligned(:,ts{3}(end)+1:ts{3}(end)+choice_tw) = ...
        data.face_rew_shift(:,ts{4}(end)-choice_tw+1:ts{4}(end));
    data.face_aligned = data.face_aligned(:,[1:ts{3}(end)+choice_tw, ts{5}(1):end]);
    % body
    data.body_aligned = data.body_rew;
    data.body_aligned(:,ts{3}(end)+1:ts{3}(end)+choice_tw) = ...
        data.body_rew_shift(:,ts{4}(end)-choice_tw+1:ts{4}(end));
    data.body_aligned = data.body_aligned(:,[1:ts{3}(end)+choice_tw, ts{5}(1):end]);
    
    data.pupil_r = data.pupil_aligned;
    data.face = data.face_aligned;
    data.body = data.body_aligned;
    
    ts{4} = ts{3}(end)+1:ts{3}(end)+choice_tw;
    ts{5} = (ts{4}(end)+1) : (ts{4}(end)+1) + length(ts{5});
    ts{6} = (ts{5}(end)+1) : size(data.pupil_r,2);
    data.trial_length = ts{6}(end);
    data.ts = ts;
    
    
    %% remove early licks 
    for i = 1:data.num_trial
        lick_frame = data.first_correct_lick(i);
        if isnan(lick_frame); continue; end
        lick_frame = max(lick_frame - 2, 1);
        idx_remove = lick_frame:ts{3}(end);
    
        data.pupil_r(i,idx_remove) = NaN;
        data.face(i,idx_remove) = NaN;
        data.body(i,idx_remove) = NaN;
    end
    
    %% load all sessions
    for m = 1:num_sess
        
        % behavior rate
        trial_vec_sess = data.trial_vec(sess_idx{m});
        n_act = sum(trial_vec_sess==1) + sum(trial_vec_sess==2) + ...
            sum(trial_vec_sess==3) + sum(trial_vec_sess==4) + ...
            sum(trial_vec_sess==7.5) + sum(trial_vec_sess==8.5)...
            + sum(trial_vec_sess==7.6) + sum(trial_vec_sess==8.6);
        n_correct = sum(trial_vec_sess==1) + sum(trial_vec_sess==2);
        beh_rate(count+m) = n_correct/n_act*100;
        
        % assemble trial indices
        t_idx = cell(1,tnum);
        for n = 1:tnum
            t_idx{n} = find(trial_vec_sess==2*n-1|trial_vec_sess==2*n);
        end
    
        % align to standard trial structure
        pupil_sess = reshape(data.pupil_r(sess_idx{m},:)', [], 1);
        pupil_sess = movmean(pupil_sess, 3);
        
        pupil_sess = zscore_nan(pupil_sess);
        pupil_sess = reshape(pupil_sess, data.trial_length, length(sess_idx{m}))';
        pupil_sess = align_trial_to_standard_2d(pupil_sess, ts, standard_ts);
    
        body_sess = data.body(sess_idx{m},:);
        body_sess = reshape(movmean(reshape(body_sess',[],1),3),data.trial_length,length(sess_idx{m}))';
        body_sess = (body_sess - nanmin(body_sess(:)))/(nanmax(body_sess(:)) - nanmin(body_sess(:)));
        body_sess = align_trial_to_standard_2d(body_sess, ts, standard_ts);
    
        face_sess = data.face(sess_idx{m},:);
        face_sess = reshape(movmean(reshape(face_sess',[],1),3),data.trial_length,length(sess_idx{m}))';
        face_sess = (face_sess - nanmin(face_sess(:)))/(nanmax(face_sess(:)) - nanmin(face_sess(:)));
        face_sess = align_trial_to_standard_2d(face_sess, ts, standard_ts);
    
        % store data
        for n = 1:tnum
            pupil{count+m,n} = pupil_sess(t_idx{n},:);
            body{count+m,n} = body_sess(t_idx{n},:);
            face{count+m,n} = face_sess(t_idx{n},:);
        end
     
    end
    count = count + num_sess;

end
fprintf('\n');

title_str = {'Pupil diameter', 'Body movement', 'Face movement'};

%% plot face movement without early licks - Fig. S1c
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

trial = 1:2;
cc = {mycc.gray, mycc.orange};
lstr = {'Naive', 'Expert'};
figure; set(gcf,'color','w');hold on;
for k = 1:2
    if k==1; idx = idx_set & beh_rate<55;
    else; idx = idx_set & beh_rate>=75;
    end
    v = face(idx,trial); yl = [0 1];
    v = mat2cell(v,ones(size(v,1),1),2);
    v = cellfun(@(x) cell2mat(x'), v, 'uniformoutput', 0);
    v = cell2mat(cellfun(@(x) nanmean((movmax(x, 3, 2)>0.09),1), v, 'uniformoutput', 0));
    ym = nanmean(v,1);  yse = nanstd(v,[],1)/sqrt(size(v,1));
    h = confplot(tvec, ym, yse, yse, cc{k}, 0.2);
end
xlim([0.1 5.5]); ylim(yl);
draw_trial_structure(standard_ts_fr(2:5));
xlabel('Time (s)'); 
title(title_str{i}, 'FontWeight', 'Normal');
ylabel('Normalized unit'); 


%% face only quantification - Fig.S1d
idx = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx = idx & sess_len>=80;

trial = [1,2];
tw = 2:5;
figure; set(gcf,'color','w'); 
pval_all = nan(length(tw), nbins-1);
pval_y = nan(length(tw), nbins-1);
for t = 1:length(tw)
    subplot(1,length(tw),t); hold on; h = []; ymi = Inf; yma = -Inf;
    cc = cc_ts{tw(t)};
    ydata = cell(nbins,1); v = face(idx,trial);
    v = mat2cell(v,ones(size(v,1),1),2);
    v = cellfun(@(x) cell2mat(x'), v, 'uniformoutput', 0);
    v = cell2mat(cellfun(@(x) nanmean((movmax(x, 3, 2)>0.09),1), v, 'uniformoutput', 0));
    ym = zeros(nbins,1); yse = zeros(nbins,1);
    for i = 1:nbins
        if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
        elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
        else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
        end
        ydata{i} = nanmean(v(bin_idx,standard_ts{tw(t)}),2);
        ym(i) = nanmean(ydata{i});
        yse(i) = nanstd(ydata{i})/sqrt(length(ydata{i}));
        if i>1
            pval = ranksum(ydata{1}, ydata{i});
            pval_all(t,i-1) = pval;
            pval_y(t,i-1) = ym(i)+yse(i);
            % pval = pval * (nbins - 1);
            % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
        end
    end
    h(end+1) = errorbar(rate_bin_center, ym, yse, ...
        'color', cc, 'CapSize', 3, 'LineWidth', 1);
    xlim([rate_bin_center(1)-10 rate_bin_center(end)+10]);
    if t==1; ylabel('Normalized unit'); end
    set(gca, 'xtick', rate_bin_center, 'xticklabel', {'Naive', 'Learning', 'Expert'},...
        'xticklabelrotation', 45);
    title(ts_str(tw(t)), 'FontWeight', 'Normal');
end
linkaxes;

% statistical test
tmp = mafdr(reshape(pval_all, [], 1));
fdr_all = reshape(tmp, size(pval_all,1), size(pval_all, 2));
for t = 1:length(tw)
    subplot(1,length(tw),t); hold on;
    for i = 2:nbins
        if fdr_all(t,i-1)<0.05
            plot_pval_star(rate_bin_center(i), pval_y(t,i-1), pval_all(t,i-1));
        end
    end
end

