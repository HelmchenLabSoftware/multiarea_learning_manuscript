

%%
clear all; clc
setup_colors;
datasheet = get_data_sheet('multiarea');

opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p';

var_to_read = {'trial_vec', 'pupil_r', 'body', 'ts', 'trial', 'face',...
    'trial_length', 'rate', 'choice_time'};

dataset = [132:256, 388:672]; 

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
    
    %% split session
    sess_len = 120;  min_sess_len = 5;
    sess_idx = split_session(num_trial, sess_len, min_sess_len);
    num_sess = length(sess_idx);
    
    for m = 1:num_sess
        sess_idx{m} = keep_idx(sess_idx{m});
    end
    
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
        pupil_sess = align_trial_to_standard_2d(pupil_sess, data.ts, standard_ts);
    
        body_sess = data.body(sess_idx{m},:);
        body_sess = reshape(movmean(reshape(body_sess',[],1),3),data.trial_length,length(sess_idx{m}))';
        body_sess = (body_sess - nanmin(body_sess(:)))/(nanmax(body_sess(:)) - nanmin(body_sess(:)));
        body_sess = align_trial_to_standard_2d(body_sess, data.ts, standard_ts);
    
        face_sess = data.face(sess_idx{m},:);
        face_sess = reshape(movmean(reshape(face_sess',[],1),3),data.trial_length,length(sess_idx{m}))';
        face_sess = (face_sess - nanmin(face_sess(:)))/(nanmax(face_sess(:)) - nanmin(face_sess(:)));
        face_sess = align_trial_to_standard_2d(face_sess, data.ts, standard_ts);
    
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

%% Fig. 1j,k, plot behavior variables
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

trial = 1:2;
cc = {mycc.gray, mycc.orange};
lstr = {'Naive', 'Expert'};
figure; set(gcf,'color','w');
for i = 1:3
    subplot(1,3,i); hold on;
    for k = 1:2
        if k==1; idx = idx_set & beh_rate<55;
        else; idx = idx_set & beh_rate>75;
        end
        if i==1; v = pupil(idx,trial); yl = [-0.5 0.5];
        elseif i==2; v = body(idx,trial); yl = [0 0.2];
        elseif i==3; v = face(idx,trial); yl = [0 0.4];
        end
        v = mat2cell(v,ones(size(v,1),1),2);
        v = cellfun(@(x) cell2mat(x'), v, 'uniformoutput', 0);
        v = cell2mat(cellfun(@(x) nanmean(x,1), v, 'uniformoutput', 0));
        ym = nanmean(v,1);  yse = nanstd(v,[],1)/sqrt(size(v,1));
        h = confplot(tvec, ym, yse, yse, cc{k}, 0.2);
    end
    xlim([0.1 5.5]); ylim(yl);
    draw_trial_structure(standard_ts_fr(2:5));
    xlabel('Time (s)'); 
    title(title_str{i}, 'FontWeight', 'Normal');
    if i==1; ylabel('Normalized unit'); end
end
% set_figure_style(gcf);


%% Fig. 1j,k, quantification by performance
idx = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx = idx & sess_len>=80;

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

trial = 1:2;
tw = 2:5;
figure; set(gcf,'color','w'); 
pval_all = nan(length(tw), nbins-1, 3);
pval_y = nan(length(tw), nbins-1, 3);
fdr_all = nan(length(tw), nbins-1, 3);
for m = 1:3
    subplot(1,3,m); hold on; h = []; ymi = Inf; yma = -Inf;
    for t = 1:length(tw)
        cc = cc_ts{tw(t)};
        ydata = cell(nbins,1);
        if m==1; v = pupil(idx,trial);
        elseif m==2; v = body(idx,trial);
        elseif m==3; v = face(idx,trial);
        end
        v = mat2cell(v,ones(size(v,1),1),2);
        v = cellfun(@(x) cell2mat(x'), v, 'uniformoutput', 0);
        v = cell2mat(cellfun(@(x) nanmean(x,1), v, 'uniformoutput', 0));
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
                % pval = pval * (nbins - 1);
                pval_all(t,i-1,m) = pval;
                pval_y(t,i-1,m) = ym(i)+yse(i);
                % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
            end
        end
        h(end+1) = errorbar(rate_bin_center, ym, yse, ...
            'color', cc, 'CapSize', 3, 'LineWidth', 1);
        ymi = min(ymi, min(ym-yse)); yma = max(yma, max(ym+yse)); 
    end
%     xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]);
    xlim([rate_bin_center(1)-10 rate_bin_center(end)+10]);
    ylim([ymi-(yma-ymi)*0.2, yma+(yma-ymi)*0.2]);
    if m==1; ylabel('Normalized unit'); end
    set(gca, 'xtick', rate_bin_center, 'xticklabel', {'Naive', 'Learning', 'Expert'},...
        'xticklabelrotation', 45);
    title(title_str{m}, 'FontWeight', 'Normal');

    % statistical test
    tmp = mafdr(reshape(pval_all(:,:,m), [], 1));
    fdr_all(:,:,m) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    for t = 1:length(tw)
        for i = 2:nbins
            if fdr_all(t,i-1,m)<0.05
                plot_pval_star(rate_bin_center(i), pval_y(t,i-1,m), pval_all(t,i-1,m));
            end
        end
    end

end
legend(h, ts_str(tw));

%% plot correct vs incorrect - Fig. S8a,c
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

trial = [1,2];
cc = {mycc.gray, mycc.orange};
lstr = {'Naive', 'Expert'};
title_str = {'Pupil diameter', 'Body movement', 'Face movement'};
figure; set(gcf,'color','w');
for i = 1:3
    subplot(1,3,i); hold on;
    for k = 1:2
        if k==1; idx = idx_set & beh_rate<55;
        else; idx = idx_set & beh_rate>75;
        end
        for n = 1:length(trial)
            if i==1; v = pupil(idx,trial(n)); yl = [-0.55 0.65];
            elseif i==2; v = body(idx,trial(n)); yl = [0 0.2];
            elseif i==3; v = face(idx,trial(n)); yl = [0 0.4];
            end
            v = cell2mat(cellfun(@(x) nanmean(x,1), v, 'uniformoutput', 0));
            ym = nanmean(v,1);  yse = nanstd(v,[],1)/sqrt(size(v,1));
            h = confplot(tvec, ym, yse, yse, cc{k}, 0.2);
            if n==2; set(h(1), 'linestyle', '--'); end
        end
    end
    xlim([0.1 5.5]); ylim(yl);
    draw_trial_structure(standard_ts_fr(2:5));
    xlabel('Time (s)'); 
    title(title_str{i}, 'FontWeight', 'Normal');
    if i==1; ylabel('Normalized unit'); end
end

%% quantification by performance binned, correct vs incorrect - Fig. S8b,d
idx = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx = idx & sess_len>=80;

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

trial = [1,2];
tw = 2:5;
figure; set(gcf,'color','w'); w = 2; mksz = 4;
pval_all = nan(length(tw), nbins, 3);
pval_y = nan(length(tw), nbins, 3);
fdr_all = nan(length(tw), nbins, 3);
for m = 1:3
    for t = 1:length(tw)
        subplot(3,length(tw),(m-1)*length(tw)+t); hold on; h = []; ymi = Inf; yma = -Inf;
        cc = cc_ts{tw(t)};
        ydata = cell(nbins,2);
        for n = 1:length(trial)
            if m==1; v = pupil(idx,trial(n));
            elseif m==2; v = body(idx,trial(n));
            elseif m==3; v = face(idx,trial(n));
            end
            v = cell2mat(cellfun(@(x) nanmean(x,1), v, 'uniformoutput', 0));
            ym = zeros(nbins,1); yse = zeros(nbins,1);
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i,n} = nanmean(v(bin_idx,standard_ts{tw(t)}),2);
                ym(i) = nanmean(ydata{i,n});
                yse(i) = nanstd(ydata{i,n})/sqrt(length(ydata{i,n}));
                if n==2
                    pval = signrank(ydata{i,1}, ydata{i,2});
                    pval_all(t,i,m) = pval;
                    pval_y(t,i,m) = ym(i)+yse(i);
                    % pval = pval * nbins;
                    % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
                end
            end
            h = errorbar(rate_bin_center, ym, yse, ...
                'color', cc, 'CapSize', 3, 'LineWidth', 1);
            if n==2; set(h, 'linestyle', '--'); end
            ymi = min(ymi, min(ym-yse)); yma = max(yma, max(ym+yse)); 
        end
        xlim([rate_bin_center(1)-10 rate_bin_center(end)+10]);
        ylim([ymi-(yma-ymi)*0.3, yma+(yma-ymi)*0.3]);
        if t==1; ylabel(title_str{m}); end
        set(gca, 'xtick', rate_bin_center, 'xticklabel', []);
        title(ts_str{tw(t)}, 'FontWeight', 'Normal');
    end

    % statistical test
    tmp = mafdr(reshape(pval_all(:,:,m), [], 1));
    fdr_all(:,:,m) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    for t = 1:length(tw)
        subplot(3,length(tw),(m-1)*length(tw)+t);
        for i = 1:nbins
            if fdr_all(t,i,m)<0.05
                plot_pval_star(rate_bin_center(i), pval_y(t,i,m), pval_all(t,i,m));
            end
        end
    end

end


%% plot face movement with early licks - Fig. S1a
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


%% face only quantification - Fig.S1b
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
    v = cell2mat(cellfun(@(x) nanmean((movmax(x, 3, 2)>0.1),1), v, 'uniformoutput', 0));
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
ylim([0.4, 1]);

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