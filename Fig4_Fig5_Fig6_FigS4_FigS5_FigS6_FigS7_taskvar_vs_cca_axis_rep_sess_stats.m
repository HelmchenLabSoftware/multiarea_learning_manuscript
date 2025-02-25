

%%
addpath('../calcium_analysis');
addpath(genpath('../multiarea_analysis'));
addpath(genpath('./'));

%%
clear; clc
setup_colors;
datasheet = get_data_sheet('multiarea');

%% 
% selected dataset
dataset = [132:256, 388:672];

opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p';
var_to_read = {'trial_vec'};

label_str = {'Tone', 'Texture', 'Choice', 'Reward'};
cond_tw = [2,3,4,5];
num_cond = length(label_str);
cca_tw = [2,3,4,5];
num_cca = length(cca_tw);
result_name = 'taskvar_vs_cca_axis_rep_sess_120_pc30';  % final version
% result_name = 'taskvar_vs_cca_axis_rep_sess_120_pc60';  % 60 PCs, final results

num_rep = 10; 
% num_rep = 20;

result_var = {'beh_rate', 'sess_idx', 'num_sess', 'S_num',...
    'var_explained', 'task_axis',...
    'pred_auc', 'pred_auc_shuff', 'svm_sim', 'ncv_sub', ...
    'cca_r_sub', 'cca_r_sub_shuff', 'ncv_within', ...
    'cca_r_within', 'cca_r_within_shuff', 'svm_cca_sim', 'svm_cca_sim_within', ...
    'cca_sim_sub', 'cca_sim_within', 'cca_sim_inter_vs_within', ...
    'svm_cca_sim_shuff', 'svm_cca_sim_within_shuff', 'cca_sim_sub_shuff', ...
    'cca_sim_within_shuff', 'cca_sim_inter_vs_within_shuff',...
    'cca_within_auc', 'cca_within_auc_shuff', ...
    'cca_inter_auc', 'cca_inter_auc_shuff'};

num_trial_type = 4;
tnum = num_trial_type/2;

% plot settings
fig_pos_2 = [813 124 564 500];
fig_pos_3 = [359 157 529 683];

%% collect data
num_area = 2;
num_shuff = 100;
num_shuff2 = 50;
N = num_rep * 2;  % considering two subsets per repeat

nset = length(dataset);
S_num = nan(nset, num_area, N);
var_explained = nan(nset, num_area, N);

pred_auc = nan(nset, num_cond, num_area, N);
pred_auc_shuff = nan(nset, num_cond, num_area, num_shuff, N);
svm_sim = nan(nset, num_cond, num_cond, num_area, N);

% all trials
ncv_inter = nan(nset, num_cca, N);
cca_r_inter = zeros(nset, num_cca, N);
ncv_within = nan(nset, num_cca, num_area, num_rep);
cca_r_within = zeros(nset, num_cca, num_area, num_rep);
cca_r_inter_shuff = zeros(nset, num_cca, num_shuff, N);
cca_r_within_shuff = zeros(nset, num_cca, num_area, num_shuff, num_rep);

cca_sim_inter = nan(nset, num_cca, num_cca, num_area, N);
cca_sim_within = nan(nset, num_cca, num_cca, num_area, N);
cca_sim_inter_vs_within = nan(nset, num_cca, num_cca, num_area, N);
svm_cca_sim = nan(nset, num_cond, num_cca, num_area, N);
svm_cca_sim_within = nan(nset, num_cond, num_cca, num_area, N);

cca_sim_inter_shuff = nan(nset, num_cca, num_cca, num_area, num_shuff, N);
cca_sim_within_shuff = nan(nset, num_cca, num_cca, num_area, num_shuff, N);
cca_sim_inter_vs_within_shuff = nan(nset, num_cca, num_cca, num_area, num_shuff, N);
svm_cca_sim_shuff = nan(nset, num_cond, num_cca, num_area, num_shuff, N);
svm_cca_sim_within_shuff = nan(nset, num_cond, num_cca, num_area, num_shuff, N);

cca_within_auc = nan(nset, num_cond, num_cca, 2, 2);
cca_within_auc_shuff = nan(nset, num_cond, num_cca, 2, num_shuff2, 2);
cca_inter_auc = nan(nset, num_cond, num_cca, 2, 2);
cca_inter_auc_shuff = nan(nset, num_cond, num_cca, 2, num_shuff2, 2);

beh_rate = nan(1,nset); 
exp_idx = nan(1,nset); 
a_idx = zeros(1,nset); 
mouse_id = nan(1,nset); 
quality_idx = nan(1,nset);
sess_len = nan(1,nset);
dataset_id = nan(1,nset);

% collect results
count = 0;
for dataid = 1:nset
    
    dinfo = data_info(datasheet, dataset(dataid), 'multiarea', opts.base_dir);
    spath = fullfile(dinfo.work_dir, opts.result_dir);
    result_file = fullfile(spath, sprintf('%s_rep_%d.mat', result_name, 1));
    if ~exist(result_file); continue; end
    
    ld = load(result_file, result_var{:});
    data = load_data(dinfo, var_to_read, opts);
    
    sess_idx = count+1:count+ld.num_sess;
    exp_idx(sess_idx) = get_exp_condition_idx(dinfo);
    if strcmp(dinfo.areas, 'A/RL'); a_idx(sess_idx) = 1; 
    elseif strcmp(dinfo.areas, 'S1/A'); a_idx(sess_idx) = 2; 
    else; a_idx(sess_idx) = 0; 
    end
    mouse_id(sess_idx) = dinfo.mouse_id;
    if length(ld.beh_rate)>length(sess_idx)
        beh_rate(sess_idx) = ld.beh_rate(1:ld.num_sess);
    else
        beh_rate(sess_idx) = ld.beh_rate;
    end
    % beh_rate(sess_idx) = ld.beh_rate;
    quality_idx(sess_idx) = dinfo.quality_idx;
    dataset_id(sess_idx) = dataset(dataid);
    sess_len(sess_idx) = cellfun(@(x) length(x), ld.sess_idx);
    
    %% store all repetition results
    for rep_idx = 1:num_rep
        
        result_file = fullfile(spath, sprintf('%s_rep_%d.mat', result_name, rep_idx));
        if ~exist(result_file); continue; end
        ld = load(result_file, result_var{:});
        
        n_idx = (rep_idx-1)*2+1:rep_idx*2;  % current subsample indices
        
        S_num(sess_idx, :, n_idx) = ld.S_num;
        var_explained(sess_idx, :, n_idx) = ld.var_explained;
        
        pred_auc(sess_idx, :, :,  n_idx) = ld.pred_auc;
        pred_auc_shuff(sess_idx, :, :, :, n_idx) = ld.pred_auc_shuff;
        svm_sim(sess_idx, :, :, :, n_idx) = abs(ld.svm_sim);

        % all trials
        ncv_inter(sess_idx, :, n_idx) = ld.ncv_sub;
        cca_r_inter(sess_idx, :, n_idx) = ld.cca_r_sub;
        cca_r_inter_shuff(sess_idx, :, :, n_idx) = permute(ld.cca_r_sub_shuff,[1,2,4,3]);
        ncv_within(sess_idx, :, :, rep_idx) = ld.ncv_within;
        cca_r_within(sess_idx, :, :, rep_idx) = ld.cca_r_within;
        cca_r_within_shuff(sess_idx, :, :, :, rep_idx) = ld.cca_r_within_shuff;

        cca_sim_inter(sess_idx, :, :, :, n_idx) = ld.cca_sim_sub;
        cca_sim_within(sess_idx, :, :, :, n_idx) = ld.cca_sim_within;
        cca_sim_inter_vs_within(sess_idx, :, :, :, n_idx) = ld.cca_sim_inter_vs_within;
        svm_cca_sim(sess_idx, :, :, :, n_idx) = ld.svm_cca_sim;
        svm_cca_sim_within(sess_idx, :, :, :, n_idx) = ld.svm_cca_sim_within;
        
        cca_sim_inter_shuff(sess_idx, :, :, :, :, n_idx) = ld.cca_sim_sub_shuff;
        cca_sim_within_shuff(sess_idx, :, :, :, :, n_idx) = ld.cca_sim_within_shuff;
        cca_sim_inter_vs_within_shuff(sess_idx, :, :, :, :, n_idx) = ld.cca_sim_inter_vs_within_shuff;
        svm_cca_sim_shuff(sess_idx, :, :, :, :, n_idx) = ld.svm_cca_sim_shuff;
        svm_cca_sim_within_shuff(sess_idx, :, :, :, :, n_idx) = ld.svm_cca_sim_within_shuff;
        
        cca_within_auc(sess_idx, :, :, :, n_idx) = ld.cca_within_auc;
        cca_within_auc_shuff(sess_idx, :, :, :, :, n_idx) = ld.cca_within_auc_shuff;
        cca_inter_auc(sess_idx, :, :, :, n_idx) = ld.cca_inter_auc;
        cca_inter_auc_shuff(sess_idx, :, :, :, :, n_idx) = ld.cca_inter_auc_shuff;

    end

    count = count + ld.num_sess;
    
end

%% take absolute value
cca_sim_inter = abs(cca_sim_inter);
cca_sim_within = abs(cca_sim_within);
cca_sim_inter_vs_within = abs(cca_sim_inter_vs_within);
svm_cca_sim = abs(svm_cca_sim);
svm_cca_sim_within = abs(svm_cca_sim_within);

cca_sim_inter_shuff = abs(cca_sim_inter_shuff);
cca_sim_within_shuff = abs(cca_sim_within_shuff);
cca_sim_inter_vs_within_shuff = abs(cca_sim_inter_vs_within_shuff);
svm_cca_sim_shuff = abs(svm_cca_sim_shuff);
svm_cca_sim_within_shuff = abs(svm_cca_sim_within_shuff);

% AUC to discrimination index
pred_auc = (pred_auc-0.5) * 2;
pred_auc_shuff = (pred_auc_shuff-0.5) * 2;
cca_within_auc = abs(cca_within_auc-0.5) * 2;
cca_within_auc_shuff = abs(cca_within_auc_shuff-0.5) * 2;
cca_inter_auc = abs(cca_inter_auc-0.5) * 2;
cca_inter_auc_shuff = abs(cca_inter_auc_shuff-0.5) * 2;

%% average all repetitions
% ***************** SD ***********************
% average repetitions
pred_auc_sd = nanstd(pred_auc, [], 4);
svm_sim_sd = nanstd(svm_sim, [], 5);

% all trials
ncv_inter_sd = nanstd(ncv_inter, [], 3);
cca_r_inter_sd = nanstd(cca_r_inter, [], 3);
ncv_within_sd = nanstd(ncv_within, [], 4);
cca_r_within_sd = nanstd(cca_r_within, [], 4);

cca_sim_inter_sd = nanstd(cca_sim_inter, [], 5);
cca_sim_within_sd = nanstd(cca_sim_within, [], 5);
cca_sim_inter_vs_within_sd = nanstd(cca_sim_inter_vs_within, [], 5);
svm_cca_sim_sd = nanstd(svm_cca_sim, [], 5);
svm_cca_sim_within_sd = nanstd(svm_cca_sim_within, [], 5);

% ***************** MEAN ***********************
% average repetitions
pred_auc = nanmean(pred_auc, 4);
pred_auc_shuff = nanmean(pred_auc_shuff, 5);
svm_sim = nanmean(svm_sim, 5);

% all trials
ncv_inter = nanmean(ncv_inter, 3);
cca_r_inter = nanmean(cca_r_inter, 3);
ncv_within = nanmean(ncv_within, 4);
cca_r_within = nanmean(cca_r_within, 4);
cca_r_inter_shuff = nanmean(cca_r_inter_shuff, 4);
cca_r_within_shuff = nanmean(cca_r_within_shuff, 4);

% average results
cca_sim_inter = nanmean(cca_sim_inter, 5);
cca_sim_within = nanmean(cca_sim_within, 5);
cca_sim_inter_vs_within = nanmean(cca_sim_inter_vs_within, 5);
svm_cca_sim = nanmean(svm_cca_sim, 5);
svm_cca_sim_within = nanmean(svm_cca_sim_within, 5);

% average shuffled results
cca_sim_inter_shuff = nanmean(cca_sim_inter_shuff, 6);
cca_sim_within_shuff = nanmean(cca_sim_within_shuff, 6);
cca_sim_inter_vs_within_shuff = nanmean(cca_sim_inter_vs_within_shuff, 6);
svm_cca_sim_shuff = nanmean(svm_cca_sim_shuff, 6); 
svm_cca_sim_within_shuff = nanmean(svm_cca_sim_within_shuff, 6);

cca_within_auc = nanmean(cca_within_auc, 5);
cca_within_auc_shuff = nanmean(cca_within_auc_shuff, 6);
cca_inter_auc = nanmean(cca_inter_auc, 5);
cca_inter_auc_shuff = nanmean(cca_inter_auc_shuff, 6);


%% plot learning curve
um = unique(mouse_id);
cc = turbo(length(um));
figure; set(gcf, 'color', 'w'); hold on; xma = 0;
for n = 1:length(um)
    idx = mouse_id==um(n);
    idx = idx & (exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70));
    idx = idx & sess_len>=80;
	v = beh_rate(idx);
    plot(v, 'color', cc(n,:), 'linewidth', 1);
    xma = max(xma, length(v));
end
plot([1 xma], 55*[1 1], 'k:');
plot([1 xma], 75*[1 1], 'k:');
ylim([30 100]); xlim([1 xma]);
xlabel('Session'); ylabel('Performance');
% set_figure_style(gcf);

%% print number of sessions per mouse per condition
% idx_set = (exp_idx==2 | ((exp_idx==3|exp_idx==4) & beh_rate>70));
idx_set = (exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70));
um = unique(mouse_id);

stat_mat = zeros(length(um),3,2);
for m = 1:length(um)
    for a = 1:2
        for n = 1:3
            idx = idx_set & mouse_id==um(m);
            if a==1; idx = idx & a_idx==0;
            elseif a==2; idx = idx & a_idx==2;
            end
            if n==1; idx = idx & beh_rate<55;
            elseif n==2; idx = idx & beh_rate>=55 & beh_rate<75;
            elseif n==3; idx = idx & beh_rate>=75;
            end
            stat_mat(m,n,a) = sum(idx);
        end
    end
end

%% Fig. S4a - variance explained
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
idx_set = idx_set & quality_idx==2;

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

lstr = {'Naive', 'Learning', 'Expert'};
figure; set(gcf,'color','w'); mksz = 4; pval_all = [];
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    subplot(3,1,a); hold on; ymi = Inf; yma = -Inf;
    ydata = cell(1,nbins); ym = nan(1,nbins); yse = nan(1,nbins); 
    v = nanmean(var_explained(idx,area_idx,:), 3);
    for i = 1:nbins
        if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
        elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
        else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
        end
        ydata{i} = v(bin_idx);
        h = boxplot(ydata{i}, 'position', rate_bin_center(i), 'width', 10, 'color', cc_area3{a,1});
        yl = setBoxStyle(h, 1);
        ymi = min(ymi, yl(1)); yma = max(yma, yl(2));
        pval = ranksum(ydata{1}, ydata{i});
        plot_pval_star(rate_bin_center(i), yl(2), pval);
        pval_all(i,a) = pval;
        fprintf('%2.1f +/- %2.1f\n', nanmean(ydata{i}), nanstd(ydata{i})/sqrt(sum(bin_idx)))
    end
    ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
    plot([rate_bin_center(1)-8 rate_bin_center(end)+8],[0.5 0.5],'k:');
    xlim([rate_bin_center(1)-8 rate_bin_center(end)+8]); ylim([ymi yma]); box off
%     ylim([0.2 1]);
    ylabel(sprintf('%s', area_str3{a}));
    set(gca, 'xtick', rate_bin_center, 'xticklabel', lstr, 'xticklabelrotation', 45);
end
linkaxes;
% set_figure_style(gcf);

%% Fig. 4c and Fig. S4b, auc three areas over learning
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
% idx_set = idx_set & quality_idx==2;

cond = [1,2,3,4];

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

lstr = {'Naive', 'Learning', 'Expert'};
figure; set(gcf,'color','w'); mksz = 4;
pval_all = nan(length(cond), nbins-1, 3);
pval_y = nan(length(cond), nbins-1, 3);
for k = 1:length(cond)
    for a = 1:3
        if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
        elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
        elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
        end
        subplot(3,length(cond),(a-1)*length(cond)+k); hold on; ymi = Inf; yma = -Inf;
        ydata = cell(1,nbins); ysdata = cell(1,nbins);
        ysdata1 = cell(1,nbins); ysdata2 = cell(1,nbins);
        ym = nan(1,nbins); yse = nan(1,nbins); 
        ysm = nan(1,nbins); ysse = nan(1,nbins); 
        ysm1 = nan(1,nbins); ysse1 = nan(1,nbins); ysm2 = nan(1,nbins); ysse2 = nan(1,nbins); 
        v = pred_auc(idx,cond(k),area_idx);
        % v = pred_auc_sd(idx,cond(k),area_idx);
        vs = squeeze(pred_auc_shuff(idx,cond(k),area_idx,:));
        v = -v; vs = -vs;
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            ysdata{i} = nanmean(vs(bin_idx,:),2);
            ysm(i) = nanmean(ysdata{i}); ysse(i) = nanstd(ysdata{i})/sqrt(sum(bin_idx));
            ysdata1{i} = quantile(vs(bin_idx,:), 0.05, 2);
            ysm1(i) = nanmean(ysdata1{i}); ysse1(i) = nanstd(ysdata1{i})/sqrt(sum(bin_idx));
            ysdata2{i} = quantile(vs(bin_idx,:), 0.95, 2);
            ysm2(i) = nanmean(ysdata2{i}); ysse2(i) = nanstd(ysdata2{i})/sqrt(sum(bin_idx));
            ydata{i} = v(bin_idx);
            ym(i) = nanmean(ydata{i}); yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
            % test against naive
            if i>1
                pval = ranksum(ydata{1}, ydata{i});
                % pval = pval * (nbins-1);
                pval_all(k,i-1,a) = pval;
                pval_y(k,i-1,a) = (ym(i)+yse(i))*1.05;
                % plot_pval_star(rate_bin_center(i), (ym(i)+yse(i))*1.05, pval);
            end
        end
        errorbar(rate_bin_center, ysm, ysse, 'color', mycc.gray, ...
            'CapSize', 4, 'linewidth', 1);
        errorbar(rate_bin_center, ysm1, ysse1, 'color', mycc.gray_light, ...
            'CapSize', 4, 'linewidth', 1);
        errorbar(rate_bin_center, ysm2, ysse2,   'color', mycc.gray_light, ...
            'CapSize', 4, 'linewidth', 1);
        errorbar(rate_bin_center, ym, yse, 'color', cc_area3{a,1}, ...
            'CapSize', 4, 'linewidth', 1);
        ymi = nanmin([ymi,ysm-ysse]); yma = nanmax([yma,ysm+ysse]);
        ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]); ylim([ymi yma]); box off
        ylim([0 1]);
        if a==1; title([label_str{cond(k)} ' axis'], 'fontweight', 'normal'); end
        if k==1; ylabel(sprintf('%s\nDI', area_str3{a})); end
        if a==3
            set(gca, 'xtick', rate_bin_center, 'xticklabel', lstr, 'xticklabelrotation', 45);
        else
            set(gca, 'xtick', []);
        end
    end
end
% set_figure_style(gcf);
linkaxes;

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:3
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    for k = 1:length(cond)
        subplot(3,length(cond),(a-1)*length(cond)+k); hold on; 
        for i = 2:nbins
            if fdr_all(k,i-1,a)<0.05
                plot_pval_star(rate_bin_center(i), pval_y(k,i-1,a), pval_all(k,i-1,a));
            end
        end
    end
end

%% Fig. 4d and Fig. S5b, Fig. S4c, svm axis similarity
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

pval_all = zeros(num_cond, num_cond, 3);
figure; set(gcf,'color','w');
set(gcf, 'position', fig_pos_3);
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx_exp = idx_set & a_idx==2; area_idx = 2;
    end
    v = cell(1,2);
    for n = 1:2
        if n==1; idx = idx_exp & beh_rate<55; lstr = 'Naive';
        elseif n==2; idx = idx_exp & beh_rate>=75;  lstr = 'Expert';
        end
        subplot(3,2,(a-1)*2+n);
        v{n} = svm_sim(idx,:,:,area_idx); cl = [0 0.6];
        % v{n} = svm_sim_sd(idx,:,:,area_idx);  cl = [0 0.2];
        imagesc(1:num_cond, 1:num_cond, squeeze(nanmean(v{n},1))); hold on;
        clim(cl);
        for i = 1:num_cond
            for j = 1:num_cond
                pval = ranksum(v{1}(:,i,j), v{n}(:,i,j));
                if pval<0.05; scatter(i,j,'r*'); end
                pval_all(i,j,a) = pval;
            end
        end
        if n==1; ylabel(sprintf('%s\nEncoding axis', area_str3{a})); end
        if a==1; title(lstr, 'FontWeight', 'Normal'); end
        if a==3; xlabel('Encoding axis'); end
        set(gca, 'xtick', 1:num_cond, 'xticklabel', label_str, 'xticklabelrotation', 45);
        set(gca, 'ytick', 1:num_cond, 'yticklabel', label_str);
    end
end
colormap(viridis);
% set_figure_style(gcf);

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:3
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
end
figure;
for a = 1:3
    subplot(3,1,a);
    smat = fdr_all(:,:,a)<0.05;
    smat = smat & pval_all(:,:,a)<0.05;
    imagesc(smat');
end

%% Fig. 4e, Fig. S4d, svm similarity quantification over learning, separate subplot
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

tw_pair = [1,2; 1,3; 1,4; 2,3; 2,4; 3,4];
ntw_pair = size(tw_pair,1);

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

figure; set(gcf,'color','w'); mksz = 3; lstr = {};
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    cc = {mycc.gray, (mycc.gray+cc_area3{a,1})/2, cc_area3{a,1}};
    for n = 1:ntw_pair
        subplot(3,ntw_pair,(a-1)*ntw_pair+n); hold on;
        ydata = cell(1,nbins); ymi = Inf; yma = -Inf;
        lstr{n} = sprintf('%s-%s', label_str{tw_pair(n,1)}, label_str{tw_pair(n,2)});
        v = svm_sim(idx,tw_pair(n,1),tw_pair(n,2),area_idx);
        ym = []; yse = [];
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            ydata{i} = v(bin_idx);
            ym(i) = nanmean(ydata{i});
            yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
            try; pval = ranksum(ydata{1}, ydata{i});
            catch ME; pval = NaN;
            end
            plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
        end
        errorbar(rate_bin_center, ym, yse, 'color', cc_area3{a,1}, ...
            'CapSize', 4, 'linewidth', 1);
        ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        xlim([rate_bin_center(1)-10 rate_bin_center(end)+10]); ylim([ymi*0.8 yma*1.2]);
        set(gca, 'xtick', []);
        if n>1; set(gca, 'ytick', []); end
    end
end
linkaxes;
% set_figure_style(gcf);


%% Fig. 5f, Fig. S4g, and Fig. S5e, - svm vs cca weight similarity matrix, within model
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;
% idx_set = idx_set & sess_len>=100;

cond = 1:4; tw_cca = 1:4;
% cond = 3; tw_cca = 2;
pval_all = nan(length(cond), length(tw_cca), 3);
figure; set(gcf,'color','w');
set(gcf, 'position', fig_pos_3);
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx_exp = idx_set & a_idx==2; area_idx = 2;
    end
    v = cell(1,2);
    for n = 1:2
        if n==1; idx = idx_exp & beh_rate<55; lstr = 'Naive';
        elseif n==2; idx = idx_exp & beh_rate>=75;  lstr = 'Expert';
        end
        subplot(3,2,(a-1)*2+n); 
        v{n} = svm_cca_sim_within(idx,cond,tw_cca,area_idx);  cl = [0, 0.5];
        % v{n} = svm_cca_sim_within_sd(idx,cond,tw_cca,area_idx);  cl = [0, 0.2];
        vm = permute(nanmean(v{n},1), [2,3,1]);
        imagesc(tw_cca, cond, vm); hold on;
        % significance with shuffled data
        vs = svm_cca_sim_within_shuff(idx,cond,tw_cca,area_idx,:);
        vs = quantile(vs, 0.95, 5);
        sig_map = zeros(length(cond), length(tw_cca));
        for i = 1:length(cond)
            for j = 1:length(tw_cca)
                try; pval = signrank(vs(:,i,j), v{n}(:,i,j), 'tail', 'left');
                catch ME; continue;
                end
                if pval<0.05; sig_map(i,j) = 1; end
                try; pval = ranksum(v{1}(:,i,j), v{n}(:,i,j));
                catch ME; continue;
                end
                if pval<0.05; scatter(tw_cca(j), cond(i), 'r*'); end
                pval_all(i,j,a) = pval;
            end
        end
        % overlay significance map
        sc = 100; mgsz = 10;
        sig_lines = find_sig_map_boundary(sig_map, sc, mgsz);
        for i = 1:length(sig_lines)
            plot(sig_lines{i}(:,2)/sc+cond(1)-0.5, sig_lines{i}(:,1)/sc+tw_cca(1)-0.5, 'color', mycc.red);
        end
        clim(cl);
        set(gca, 'xtick', tw_cca, 'xticklabel', ts_str(cca_tw(tw_cca)), 'xticklabelrotation', 45);
        set(gca, 'ytick', cond, 'yticklabel', label_str(cond));
        if n==1; ylabel(sprintf('%s\nEncoding axis', area_str3{a})); end
        if a==3; xlabel('Within-area axis'); end
        if a==1; title(lstr, 'FontWeight', 'Normal'); end
    end
end
colormap(viridis);
% set_figure_style(gcf);

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:3
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
end
figure;
for a = 1:3
    subplot(3,1,a); 
    smat = fdr_all(:,:,a)<0.05;
    smat = smat & pval_all(:,:,a)<0.05;
    imagesc(smat);
end


%% Fig. 5f, Fig. S5g, svm vs cca quantification over learning, single plot, within model
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

cond = 1:4; tw_cca = 1:4;

rate_bin = [55, 75];
rate_bin_center = [-0.4, 0, 0.4];
nbins = length(rate_bin_center);
cc = cc_ts(cca_tw(tw_cca))';

pval_all = zeros(length(cond), length(tw_cca), 3);
pval_y = zeros(length(cond), length(tw_cca), 3);
pval_x = zeros(length(cond), length(tw_cca), 3);
figure; set(gcf,'color','w');
set(gcf, 'position', [308 14 622 526]);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    subplot(3,1,a); hold on; lstr = {}; x = 0;
    for k = 1:length(cond)
        for tw = 1:length(tw_cca)
            x = x + 1;
            lstr{x} = sprintf('%s-%s', label_str{cond(k)}, ts_str{cca_tw(tw_cca(tw))});
            v = svm_cca_sim_within(idx,cond(k),tw_cca(tw),area_idx);
%             v = svm_cca_sim_within_sd(idx,cond(k),tw_cca(tw),area_idx);
            vs = svm_cca_sim_within_shuff(idx,cond(k),tw_cca(tw),area_idx,:);
            vs = quantile(vs, 0.95, 5);
            ydata = cell(1,nbins); ym = zeros(1,nbins); yse = zeros(1,nbins);
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i} = v(bin_idx);
                ym(i) = nanmean(ydata{i}); 
                yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
                ysm(i) = nanmean(vs(bin_idx)); 
                ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
                if i==nbins
                    pval = ranksum(ydata{1}, ydata{i});
    %                 pval = pval * (nbins-1);
                    % plot_pval_star(rate_bin_center(i) + x, ym(i)+yse(i), pval);
                    pval_all(k,tw,a) = pval;
                    pval_y(k,tw,a) = ym(i)+yse(i);
                    pval_x(k,tw,a) = rate_bin_center(i) + x;
                end
            end
            errorbar(rate_bin_center + x, ysm, ysse, 'color', mycc.gray, 'CapSize', 2, 'linewidth', 1);
            errorbar(rate_bin_center + x, ym, yse, 'color', cc{tw}, 'CapSize', 2, 'linewidth', 1);
        end
        if tw==1; ylabel(area_str3{a}); end
        set(gca, 'xtick', []);
    end
end
linkaxes;
% set_figure_style(gcf);

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:3
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    subplot(3,1,a); hold on; 
    for k = 1:length(cond)
        for tw = 1:length(tw_cca)
            if fdr_all(k,tw,a)<0.05
                plot_pval_star(pval_x(k,tw,a), pval_y(k,tw,a), pval_all(k,tw,a));
            end
        end
    end
end


%% Fig 6e-f, Fig. S4h-i, and Fig. S5f-g, svm vs cca weight similarity matrix, inter
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
% idx_set = idx_set & a_idx==0; area_pair = [1,2];
idx_set = idx_set & a_idx==2; area_pair = [1,3];

cond = [1,2,3,4]; tw_cca = 1:4;

pval_all = zeros(length(cond), length(tw_cca), 2);
figure; set(gcf,'color','w');
set(gcf, 'position', fig_pos_2);
for area_idx = 1:2
    v = cell(1,2);
    for n = 1:2
        if n==1; idx = idx_set & beh_rate<55; lstr = 'Naive';
        elseif n==2; idx = idx_set & beh_rate>=75;  lstr = 'Expert';
        end
        % if n==2; idx = idx & dataset_id~=565; end
        subplot(2,2,(area_idx-1)*2+n); 
        v{n} = svm_cca_sim(idx,cond,tw_cca,area_idx);  cl = [0, 0.35];
        % v{n} = svm_cca_sim_sd(idx,cond,tw_cca,area_idx);  cl = [0, 0.2];
        vm = squeeze(nanmean(v{n},1));
        % plot
        imagesc(tw_cca, cond, vm); hold on;
        % significance with shuffled data
        vs = svm_cca_sim_shuff(idx,cond,tw_cca,area_idx,:);
        vs = quantile(vs, 0.95, 5);
        sig_map = zeros(length(cond), length(tw_cca));
        for i = 1:length(cond)
            for j = 1:length(tw_cca)
                pval = signrank(vs(:,i,j), v{n}(:,i,j), 'tail', 'left');
                if pval<0.05; sig_map(i,j) = 1; end
                pval = ranksum(v{1}(:,i,j), v{n}(:,i,j));
                if pval<0.05; scatter(tw_cca(j), cond(i), 'r*'); end
                pval_all(i,j,area_idx) = pval;
            end
        end
        % overlay significance map
        sc = 100; mgsz = 10; 
        sig_lines = find_sig_map_boundary(sig_map, sc, mgsz);
        for i = 1:length(sig_lines)
            plot(sig_lines{i}(:,2)/sc+tw_cca(1)-0.5, sig_lines{i}(:,1)/sc+cond(1)-0.5, 'color', mycc.red);
        end
        clim(cl);
        set(gca, 'xtick', tw_cca, 'xticklabel', ts_str(cca_tw(tw_cca)), 'xticklabelrotation', 45);
        set(gca, 'ytick', cond, 'yticklabel', label_str(cond));
        if n==1; ylabel(sprintf('%s\nEncoding axis', area_str3{area_pair(area_idx)})); end
        if area_idx==1; title(lstr, 'FontWeight', 'Normal'); end
        if area_idx==2; xlabel('Inter-area axis'); end
    end
end
colormap(viridis);
% set_figure_style(gcf);

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:2
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
end
figure;
for a = 1:2
    subplot(2,1,a);
    smat = fdr_all(:,:,a)<0.05;
    smat = smat & pval_all(:,:,a)<0.05;
    imagesc(smat);
end

%% Fig.6e-f, Fig. S4h-i, svm vs cca quantification over learning, single plot, inter
idx = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx = idx & sess_len>=80;
% idx = idx & a_idx==0; area_pair = [1,2];
idx = idx & a_idx==2; area_pair = [1,3];

cond = 1:4; tw_cca = 1:4;

rate_bin = [55, 75];
rate_bin_center = [-0.35, 0, 0.35];
nbins = length(rate_bin_center);
cc = cc_ts(cca_tw(tw_cca));

figure; set(gcf,'color','w');
set(gcf, 'position', [308 14 622 526]);
pval_all = zeros(length(cond), length(tw_cca), 2);
pval_y = zeros(length(cond), length(tw_cca), 2);
pval_x = zeros(length(cond), length(tw_cca), 2);
for a = 1:2
    subplot(2,1,a); hold on; x = 0; lstr = {};
    for k = 1:length(cond)
        for tw = 1:length(tw_cca)
            x = x + 1;
            lstr{x} = sprintf('%s axis', ts_str{cca_tw(tw_cca(tw))});
            v = svm_cca_sim(idx,cond(k),tw_cca(tw),a);
            vs = svm_cca_sim_shuff(idx,cond(k),tw_cca(tw),area_idx,:);
            vs = quantile(vs, 0.95, 5);
            ym = []; yse = []; ydata = cell(1,nbins);
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                % if i==nbins;  bin_idx = bin_idx & dataset_id(idx)~=565; end
                ydata{i} = v(bin_idx);
                ym(i) = nanmean(ydata{i});
                yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
                ysm(i) = nanmean(vs(bin_idx));
                ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
                if i==nbins
                    pval = ranksum(ydata{1}, ydata{i});
                    % pval = pval * (nbins-1);
                    % plot_pval_star(rate_bin_center(i) + x, ym(i)+yse(i), pval);
                    pval_all(k,tw,a) = pval;
                    pval_y(k,tw,a) = ym(i)+yse(i);
                    pval_x(k,tw,a) = rate_bin_center(i) + x;
                end
            end
            errorbar(x + rate_bin_center, ysm, ysse, ...
                'color', mycc.gray, 'CapSize', 2, 'linewidth', 0.5);
            errorbar(x + rate_bin_center, ym, yse, ...
                'color',cc{tw}, 'CapSize', 2, 'linewidth', 0.5);
        end
    end
    set(gca, 'xtick', []);
    ylabel(area_str3{area_pair(a)});
end
% set_figure_style(gcf);

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:2
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    subplot(2,1,a); hold on; 
    for k = 1:length(cond)
        for tw = 1:length(tw_cca)
            if fdr_all(k,tw,a)<0.05
                plot_pval_star(pval_x(k,tw,a), pval_y(k,tw,a), pval_all(k,tw,a));
            end
        end
    end
end



%% Fig. 5c-d, Fig. S4e, plot within area number of significant cvs/top correlation
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

cc = {mycc.gray, mycc.pink; mycc.gray, mycc.blue; mycc.gray, mycc.green};

figure; set(gcf, 'color', 'w'); hold on; mksz = 4; 
set(gcf, 'position', [664 326 804 256]);
pval_all = nan(num_cca, 2, 3);
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    else; idx_exp = idx_set & a_idx==2; area_idx = 2;
    end
    for k = 1:2
        subplot(2,3,(k-1)*3+a); hold on; v = cell(1,2);
        for n = 1:2
            if n==1; idx = idx_exp & beh_rate<55;
            else; idx = idx_exp & beh_rate>75;
            end
            if k==1
                v{n} = cca_r_within(idx,:,area_idx);  lstr = 'Canonical corr.'; yl = [0.4 0.9];
                % v{n} = cca_r_within_sd(idx,:,area_idx);  lstr = 'Canonical corr.';
            elseif k==2
                v{n} = ncv_within(idx,:,area_idx);  lstr = 'Significant dim.'; yl = [0 6];
                % v{n} = ncv_within_sd(idx,:,area_idx);  lstr = 'Significant dim.';
            end
            ym = nanmean(v{n}, 1); yse = nanstd(v{n}, [], 1)/sqrt(sum(idx));
            errorbar(1:num_cca, ym, yse, 'color',cc{a,n}, 'CapSize', 4, 'LineWidth', 1);
        end
        for i = 1:num_cca
            try; pval = ranksum(v{1}(:,i), v{2}(:,i));
            catch ME; pval = NaN;
            end
            pval_all(i,k,a) = pval;
            plot_pval_star(i, ym(i)+yse(i), pval);
        end
        xlim([0.5 num_cca+0.5]);  ylim(yl);
        set(gca, 'xtick', 1:num_cca, 'xticklabel', ts_str(cca_tw), 'xticklabelrotation', 45);
        if a==1; ylabel(lstr); end
        if k==1; title(area_str3{a}, 'fontweight', 'normal'); end
    end
end
legend('Naive', 'Expert');

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:3
    for k = 1:2
        fdr_all(:,k,a) = mafdr(pval_all(:,k,a));
    end
end

%% Fig. 6b-c, Fig. S4f, plot cross area number of significant cvs/top correlation
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

cc = {mycc.gray, mycc.blue; mycc.gray, mycc.green};

figure; set(gcf, 'color', 'w'); hold on; mksz = 5; w = 0.2; x_seq = [-0.15 0.15];
set(gcf, 'position', [655 326 600 256]);
pval_all = nan(num_cca, 2, 2);
for a = 1:2
    if a==1; idx_exp = idx_set & a_idx==0; tstr = 'S1-RL';
    else; idx_exp = idx_set & a_idx==2;  tstr = 'S1-A';
    end
    for k = 1:2
        subplot(2,2,(k-1)*2+a); hold on; v = cell(1,2); ymi = Inf; yma = -Inf;
        for n = 1:2
            if n==1; idx = idx_exp & beh_rate<55;
            else; idx = idx_exp & beh_rate>=75;
            end
            if k==1
            v{n} = cca_r_inter(idx,:); lstr = 'Canonical corr.'; yl = [0.3 0.8];
    %         v{n} = cca_r_inter_sd(idx,:); lstr = 'Canonical corr.';
            elseif k==2
                v{n} = ncv_inter(idx,:);  lstr = 'Significant dim.'; yl = [0 2];
    %         v{n} = ncv_inter_sd(idx,:);  lstr = 'Significant dim.';
            end
            ym = nanmean(v{n}, 1); yse = nanstd(v{n}, [], 1)/sqrt(sum(idx));
            errorbar(1:num_cca, ym, yse,  'color',cc{a,n}, 'CapSize', 4, 'LineWidth', 1);
        end
        for i = 1:num_cca
            try; pval = ranksum(v{1}(:,i), v{n}(:,i));
            catch ME; pval = NaN;
            end
            pval_all(i,k,a) = pval;
            plot_pval_star(i, ym(i)+yse(i), pval);
        end
        xlim([0.5 num_cca+0.5]);  ylim(yl);
        set(gca, 'xtick', 1:num_cca, 'xticklabel', ts_str(cca_tw), 'xticklabelrotation', 45);
        if a==1; ylabel(lstr); end
        if k==1; title(tstr, 'fontweight', 'normal'); end
    end
end
% set_figure_style(gcf);
legend('Naive', 'Expert');
% linkaxes;

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:2
    for k = 1:2
        fdr_all(:,k,a) = mafdr(pval_all(:,k,a));
    end
end

%% Fig. S5a, auc three areas over learning - STD
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

cond = [1,2,3,4];

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

lstr = {'Naive', 'Learning', 'Expert'};
figure; set(gcf,'color','w'); mksz = 3;
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    subplot(3,1,a); hold on; ymi = Inf; yma = -Inf;
    for k = 1:length(cond)
        ym = nan(1,nbins); yse = nan(1,nbins); 
        v = pred_auc_sd(idx,cond(k),area_idx);
        vs = squeeze(pred_auc_shuff(idx,cond(k),area_idx,:));
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            ydata{i} = v(bin_idx);
            ym(i) = nanmean(ydata{i}); yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
        end
        errorbar(rate_bin_center, ym, yse, ...
            'color', cc_ts{cond_tw(cond(k))}, 'CapSize', 4);
        ymi = nanmin([ymi,ysm-ysse]); yma = nanmax([yma,ysm+ysse]);
        ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]); ylim([ymi yma]); box off
        ylim([0 0.2]);
        ylabel(sprintf('%s\nDI', area_str3{a}));
        set(gca, 'xtick', rate_bin_center, 'xticklabel', []);
    end
end
% set_figure_style(gcf);
linkaxes;


%% Fig. S5c, plot within area top correlation std
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

cc = {mycc.gray, mycc.pink; mycc.gray, mycc.blue; mycc.gray, mycc.green};

figure; set(gcf, 'color', 'w'); hold on; mksz = 4; 
set(gcf, 'position', [664 326 804 256]);
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    else; idx_exp = idx_set & a_idx==2; area_idx = 2;
    end
    subplot(1,3,a); hold on; v = cell(1,2);
    for n = 1:2
        if n==1; idx = idx_exp & beh_rate<55;
        else; idx = idx_exp & beh_rate>75;
        end
        v{n} = cca_r_within_sd(idx,:,area_idx);  lstr = 'Canonical corr.';
        ym = nanmean(v{n}, 1); yse = nanstd(v{n}, [], 1)/sqrt(sum(idx));
        errorbar(1:num_cca, ym, yse, 'color',cc{a,n}, 'CapSize', 4, 'LineWidth', 1);
    end
    xlim([0.5 num_cca+0.5]); ylim([0.02, 0.06]);
    set(gca, 'xtick', 1:num_cca, 'xticklabel', ts_str(cca_tw), 'xticklabelrotation', 45);
    if a==1; ylabel(lstr); end
    if k==1; title(area_str3{a}, 'fontweight', 'normal'); end
end

%% Fig. S5d, plot inter area top correlation std
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

cc = {mycc.gray, mycc.blue; mycc.gray, mycc.green};

figure; set(gcf, 'color', 'w'); hold on; mksz = 5; w = 0.2; x_seq = [-0.15 0.15];
set(gcf, 'position', [655 326 600 256]);
for a = 1:2
    if a==1; idx_exp = idx_set & a_idx==0; tstr = 'S1-RL';
    else; idx_exp = idx_set & a_idx==2;  tstr = 'S1-A';
    end
    subplot(1,2,a); hold on; v = cell(1,2); ymi = Inf; yma = -Inf;
    for n = 1:2
        if n==1; idx = idx_exp & beh_rate<55;
        else; idx = idx_exp & beh_rate>=75;
        end
        v{n} = cca_r_inter_sd(idx,:); lstr = 'Canonical corr.';
        ym = nanmean(v{n}, 1); yse = nanstd(v{n}, [], 1)/sqrt(sum(idx));
        errorbar(1:num_cca, ym, yse,  'color',cc{a,n}, 'CapSize', 4, 'LineWidth', 1);
    end
    
    xlim([0.5 num_cca+0.5]);  ylim([0.02 0.08]);
    set(gca, 'xtick', 1:num_cca, 'xticklabel', ts_str(cca_tw), 'xticklabelrotation', 45);
    if a==1; ylabel(lstr); end
    if k==1; title(tstr, 'fontweight', 'normal'); end
end



%% Fig. S7a, cca within model weight similarity
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

pval_all = zeros(num_cca, num_cca, 3);
figure; set(gcf,'color','w');
set(gcf, 'position', fig_pos_3);
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx_exp = idx_set & a_idx==2; area_idx = 2;
    end
    v = cell(1,2);
    for n = 1:2
        if n==1; idx = idx_exp & beh_rate<55; lstr = 'Naive';
        elseif n==2; idx = idx_exp & beh_rate>=75;  lstr = 'Expert';
        end
        subplot(3,2,(a-1)*2+n); 
        v{n} = cca_sim_within(idx,:,:,area_idx); % MEAN
        h = imagesc(squeeze(abs(nanmean(v{n},1)))); hold on;
        % significance with shuffled data
        vs = quantile(cca_sim_within_shuff(idx,:,:,area_idx,:), 0.95, 5);
        sig_map = zeros(num_cca, num_cca);
        for i = 1:num_cca
            for j = 1:num_cca
                try; pval = signrank(vs(:,i,j), v{n}(:,i,j), 'tail', 'left');
                catch ME; pval = NaN;
                end
                if pval<0.05; sig_map(i,j) = 1; end
                pval = ranksum(v{1}(:,i,j), v{n}(:,i,j));
                if pval<0.05; scatter(i,j,'r*'); end
                pval_all(i,j,a) = pval;
            end
        end
        % overlay significance map
        sc = 100; mgsz = 10;
        sig_lines = find_sig_map_boundary(sig_map, sc, mgsz);
        for i = 1:length(sig_lines)
            plot(sig_lines{i}(:,2)/sc+0.5, sig_lines{i}(:,1)/sc+0.5, 'color', mycc.red);
        end
        caxis([0 0.7]);
        if n==1; ylabel(sprintf('%s\nWithin-area axis', area_str3{a})); end
        if a==1; title(lstr, 'FontWeight', 'Normal'); end
        if a==3; xlabel('Within-area axis'); end
        set(gca, 'xtick', 1:num_cca, 'xticklabel', ts_str(cca_tw), 'xticklabelrotation', 45);
        set(gca, 'ytick', 1:num_cca, 'yticklabel', ts_str(cca_tw));
    end
end
colormap(viridis);
% set_figure_style(gcf);

% fdr control
fdr_all = nan(size(pval_all));
for a = 1:3
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
end
figure;
for a = 1:3
    subplot(3,1,a);
    smat = fdr_all(:,:,a)<0.05;
    smat = smat & pval_all(:,:,a)<0.05;
    imagesc(smat);
end


%% Fig. S7a, cca within weight similarity quantification over learning, separate subplots
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

tw_pair = [1,2; 1,3; 1,4; 2,3; 2,4; 3,4];
ntw_pair = size(tw_pair,1);

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

figure; set(gcf,'color','w'); mksz = 4;
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    for n = 1:ntw_pair
        subplot(3,ntw_pair,(a-1)*ntw_pair+n); hold on;
        ydata = cell(1,nbins); ymi = Inf; yma = -Inf;
%         lstr{n} = sprintf('%s-%s', label_str{tw_pair(n,1)}, label_str{tw_pair(n,2)});
        v = cca_sim_within(idx,tw_pair(n,1),tw_pair(n,2),area_idx);
        vs = cca_sim_within_shuff(idx,tw_pair(n,1),tw_pair(n,2),area_idx,:);
        vs = quantile(vs, 0.95, 5);
        ym = []; yse = [];  ysm = []; ysse = [];
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            ydata{i} = v(bin_idx);
            ym(i) = nanmean(ydata{i});
            yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
            ysm(i) = nanmean(vs(bin_idx));
            ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
            try; pval = ranksum(ydata{1}, ydata{i});
            catch ME; pval = NaN;
            end
            plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
        end
        errorbar(rate_bin_center, ysm, ysse, 'color', mycc.gray, 'CapSize', 4, 'linewidth', 1);
        errorbar(rate_bin_center, ym, yse, 'color', cc_area3{a,1}, 'CapSize', 4, 'linewidth', 1);
        ymi = nanmin([ymi,ysm-ysse]); yma = nanmax([yma,ysm+ysse]);
        ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        xlim([rate_bin_center(1)-10 rate_bin_center(end)+10]); ylim([ymi*0.8 yma*1.2]);
        set(gca, 'xtick', []);
        if n>1; set(gca, 'ytick', []); end
%         if a==1; ylim([0.25 0.65]);
%         elseif a==2; ylim([0.2 0.55]);
%         elseif a==3; ylim([0.4 0.8]);
%         end
    end
end
linkaxes;
% set_figure_style(gcf);

%% Fig. S7b-c, cca inter model weight similarity
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
% idx_set = idx_set & a_idx==0;  area_pair = [1,2];
idx_set = idx_set & a_idx==2;  area_pair = [1,3];


pval_all = zeros(num_cca, num_cca, 3);
figure; set(gcf,'color','w');
set(gcf, 'position', fig_pos_2);
    
for a = 1:2
    v0 = cell(1,2);
    for n = 1:2
        if n==1; idx = idx_set & beh_rate<55; lstr = 'Naive';
        elseif n==2; idx = idx_set & beh_rate>=75;  lstr = 'Expert';
        end
        subplot(2,2,(a-1)*2+n);
        v0{n} = cca_sim_inter(idx,:,:,a);
%         v0 = cca_sim_tt_inter(idx,:,:,1,a);
%         v0 = nanmean(cca_sim_inter_shuff(idx,:,:,area_idx,:),5);
        v = squeeze(nanmean(v0{n},1));
        imagesc(v); hold on;
        % significance with shuffled data
        vs = quantile(cca_sim_inter_shuff(idx,:,:,a,:), 0.95, 5);
%         vs = quantile(cca_sim_tt_inter_shuff(idx,tw,tw,1,a,:), 0.95, 6);
        % significance test
        sig_map = zeros(num_cca, num_cca);
        for i = 1:num_cca
            for j = 1:num_cca
                pval = signrank(vs(:,i,j), v0{n}(:,i,j), 'tail', 'left');
                if pval<0.05; sig_map(i,j) = 1; end
                pval = ranksum(v0{1}(:,i,j), v0{n}(:,i,j));
                if pval<0.05
                    if nanmean(v0{1}(:,i,j))>nanmean(v0{n}(:,i,j))
                        scatter(i,j,'k*'); 
                    else
                        scatter(i,j,'r*'); 
                    end
                end
                pval_all(i,j,a) = pval;
            end
        end
        % overlay significance map
        sc = 100; mgsz = 10;
        sig_lines = find_sig_map_boundary(sig_map, sc, mgsz);
        for i = 1:length(sig_lines)
            plot(sig_lines{i}(:,2)/sc+0.5, sig_lines{i}(:,1)/sc+0.5, 'color', mycc.red);
        end
        % end of overlay significance map
        caxis([0 0.5]);
        if n==1; ylabel(sprintf('%s\nInter-area axis', area_str3{area_pair(a)})); end
        if a==1; title(lstr, 'FontWeight', 'Normal'); end
        if a==2; xlabel('Inter-area axis'); end
        set(gca, 'xtick', 1:num_cca, 'xticklabel', ts_str(cca_tw), 'xticklabelrotation', 45);
        set(gca, 'ytick', 1:num_cca, 'yticklabel', ts_str(cca_tw));
    end
end
% set_figure_style(gcf);
colormap(viridis);

% fdr control
fdr_all = nan(size(pval_all));
for a = 1:2
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
end
figure;
for a = 1:2
    subplot(2,1,a);
    smat = fdr_all(:,:,a)<0.05;
    smat = smat & pval_all(:,:,a)<0.05;
    imagesc(smat);
end


%% Fig. S7b-c, cca inter weight similarity quantification over learning, separate plots
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
idx = idx_set & a_idx==0; area_pair = [1,2];
% idx = idx_set & a_idx==2; area_pair = [1,3];

tw_pair = nchoosek(1:num_cca,2);
ntw = size(tw_pair, 1);

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
N = (1+num_cca-1)/2*(num_cca-1);

figure; set(gcf,'color','w'); mksz = 4;
for area_idx = 1:2
    cc = cc_area3{area_pair(area_idx)};
    for n = 1:ntw
        subplot(2,ntw,(area_idx-1)*ntw+n); hold on;
        ydata = cell(1,nbins); ymi = Inf; yma = -Inf;
        v = cca_sim_inter(idx,tw_pair(n,1),tw_pair(n,2),area_idx);
        vs = cca_sim_inter_shuff(idx,tw_pair(n,1),tw_pair(n,2),area_idx,:);
        vs = quantile(vs, 0.95, 6);
        ym = []; yse = []; ysm = []; ysse = [];
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            ydata{i} = v(bin_idx);
            ym(i) = nanmean(ydata{i});
            yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
            ysm(i) = nanmean(vs(bin_idx));
            ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
            try; pval = ranksum(ydata{1}, ydata{i});
            catch ME; pval = NaN;
            end
            pval = pval * (nbins-1);
            plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
        end
        errorbar(rate_bin_center, ysm, ysse, 'color',mycc.gray, 'CapSize', 4, 'linewidth', 1);
        errorbar(rate_bin_center, ym, yse, 'color',cc, 'CapSize', 4, 'linewidth', 1);
        ymi = nanmin([ymi,ysm-ysse]); yma = nanmax([yma,ysm+ysse]);
        ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        xlim([rate_bin_center(1)-7 rate_bin_center(end)+7]); 
        ylim([ymi*0.8 yma*1.2]);
        set(gca, 'xtick', []);
        if area_idx==1; title(sprintf('%s\n%s', ts_str{cca_tw(tw_pair(n,1))}, ...
                ts_str{cca_tw(tw_pair(n,2))}), 'fontweight', 'normal'); end
        if n==1; ylabel(area_str3{area_pair(area_idx)}); end
        if n>1; set(gca, 'ytick', []); end
    end
end
linkaxes;
% set_figure_style(gcf);
% legend(lstr);



%% Fig. 6h, AUC of CCA axis projection, inter
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
idx_set = idx_set & a_idx==0;  area_pair = [1,2];
% idx_set = idx_set & a_idx==2;  area_pair = [1,3];

cond = 1:4;
tw_cca = 1:4;

pval_all = zeros(length(cond), length(tw_cca), 2);
figure; set(gcf,'color','w');
set(gcf, 'position', fig_pos_2);
for a = 1:2
    v0 = cell(1,2);
    for n = 1:2
        if n==1; idx = idx_set & beh_rate<55; lstr = 'Naive';
        elseif n==2; idx = idx_set & beh_rate>=75;  lstr = 'Expert';
        end
        subplot(2,2,(a-1)*2+n);
        v0{n} = cca_inter_auc(idx,cond,tw_cca,a);
        v = squeeze(nanmean(v0{n},1));
        imagesc(tw_cca, cond, v); hold on;
        % significance with shuffled data
        vs = cca_inter_auc_shuff(idx,:,:,a,:);
        vs = quantile(vs, 0.95, 5);
        sig_map = zeros(length(cond), length(tw_cca));
        for i = 1:length(cond)
            for j = 1:length(tw_cca)
                try; pval = signrank(vs(:,i,j), v0{n}(:,i,j), 'tail', 'left');
                catch ME; pval = NaN;
                end
                if pval<0.05; sig_map(i,j) = 1; end
                pval = ranksum(v0{1}(:,i,j), v0{n}(:,i,j));
                if pval<0.05; scatter(tw_cca(j), cond(i),'r*'); end
                pval_all(i,j,a) = pval;
            end
        end
        % overlay significance map
        sc = 100; mgsz = 10;
        sig_lines = find_sig_map_boundary(sig_map, sc, mgsz);
        for i = 1:length(sig_lines)
            plot(sig_lines{i}(:,2)/sc+0.5, sig_lines{i}(:,1)/sc+0.5, 'color', mycc.red);
        end
        caxis([0 0.6]);
        if n==1; ylabel(sprintf('%s\nDecoder variable', area_str3{area_pair(a)})); end
        if a==1; title(lstr, 'FontWeight', 'Normal'); end
        if a==2; xlabel('Inter-area axis'); end
        set(gca, 'xtick', tw_cca, 'xticklabel', ts_str(cca_tw(tw_cca)), 'xticklabelrotation', 45);
        set(gca, 'ytick', cond, 'yticklabel', label_str);
    end
end
colormap(viridis);
% set_figure_style(gcf);

% fdr control
fdr_all = nan(size(pval_all));
for a = 1:2
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
end
figure;
for a = 1:2
    subplot(2,1,a);
    smat = fdr_all(:,:,a)<0.05;
    smat = smat & pval_all(:,:,a)<0.05;
    imagesc(smat);
end

%% Fig. 6h,  cca AUC inter quantification over learning, single plot
idx = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx = idx & a_idx==0; area_pair = [1,2];
idx = idx & a_idx==2; area_pair = [1,3];

cond = 1:4; tw_cca = 1:4;

rate_bin = [55, 75];
rate_bin_center = [-0.35, 0, 0.35];
cc = cc_ts(cca_tw(tw_cca));

figure; set(gcf,'color','w');
set(gcf, 'position', [308 14 622 526]);
for a = 1:2
    subplot(2,1,a); hold on; x = 0; lstr = {};
    for k = 1:length(cond)
        for tw = 1:length(tw_cca)
            x = x + 1;
            lstr{x} = sprintf('%s axis', ts_str{cca_tw(tw_cca(tw))});
            v = cca_inter_auc(idx,cond(k),tw_cca(tw),a);
            vs = cca_inter_auc_shuff(idx,cond(k),tw_cca(tw),area_idx,:);
            vs = quantile(vs, 0.95, 5);
            ym = []; yse = []; ydata = cell(1,nbins);
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i} = v(bin_idx);
                ym(i) = nanmean(ydata{i});
                yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
                ysm(i) = nanmean(vs(bin_idx));
                ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
                pval = ranksum(ydata{1}, ydata{i});
%                 pval = pval * (nbins-1);
                plot_pval_star(x + rate_bin_center(i), ym(i)+yse(i), pval);
            end
            errorbar(x + rate_bin_center, ysm, ysse, ...
                'color', mycc.gray, 'CapSize', 2, 'linewidth', 0.5);
            errorbar(x + rate_bin_center, ym, yse, ...
                'color',cc{tw}, 'CapSize', 2, 'linewidth', 0.5);
        end
    end
    set(gca, 'xtick', []);
    ylabel(area_str3{area_pair(a)});
end
% set_figure_style(gcf);


%% Fig. 5h, AUC of CCA axis projection, within
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

cond = 1:4;
tw_cca = 1:4;

pval_all = zeros(length(cond), length(tw_cca), 3);
figure; set(gcf,'color','w');
set(gcf, 'position', fig_pos_3);
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx_exp = idx_set & a_idx==2; area_idx = 2;
    end
    v = cell(1,2);
    for n = 1:2
        if n==1; idx = idx_exp & beh_rate<55; lstr = 'Naive';
        elseif n==2; idx = idx_exp & beh_rate>=75;  lstr = 'Expert';
        end
        subplot(3,2,(a-1)*2+n); 
        v{n} = cca_within_auc(idx,cond,tw_cca,area_idx);
        vm = squeeze(nanmean(v{n},1));
        imagesc(vm); hold on;
        % significance with shuffled data
        vs = cca_within_auc_shuff(idx,cond,tw_cca,area_idx,:);
        vs = quantile(vs, 0.95, 5);
        sig_map = zeros(num_cond, num_cca);
        for i = 1:length(cond)
            for j = 1:length(tw_cca)
                try; pval = signrank(vs(:,i,j), v{n}(:,i,j), 'tail', 'left');
                catch ME; continue;
                end
                if pval<0.05; sig_map(i,j) = 1; end
                pval = ranksum(v{1}(:,i,j), v{n}(:,i,j));
                if pval<0.05; scatter(tw_cca(j), cond(i), 'r*'); end
                pval_all(i,j,a) = pval;
            end
        end
        % overlay significance map
        sc = 100; mgsz = 10;
        sig_lines = find_sig_map_boundary(sig_map, sc, mgsz);
        for i = 1:length(sig_lines)
            plot(sig_lines{i}(:,2)/sc+tw_cca(1)-0.5, sig_lines{i}(:,1)/sc+cond(1)-0.5, 'color', mycc.red);
        end
        caxis([0 0.6]);
        set(gca, 'xtick', tw_cca, 'xticklabel', ts_str(cca_tw(tw_cca)), 'xticklabelrotation', 45);
        set(gca, 'ytick', cond, 'yticklabel', label_str(cond));
        if n==1; ylabel(sprintf('%s\nDecoder variable', area_str3{a}));  end
        if a==1; title(lstr, 'FontWeight', 'Normal'); end
        if a==3; xlabel('Within-area axis'); end
    end
end
colormap(viridis);
% set_figure_style(gcf);

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:3
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
end
figure;
for a = 1:3
    subplot(3,1,a);
    smat = fdr_all(:,:,a)<0.05;
    smat = smat & pval_all(:,:,a)<0.05;
    imagesc(smat);
end


%% Fig. 5h, AUC of CCA axis projection, within quantification, single plot
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

cond = 1:4; tw_cca = 1:4;
N = length(cond)*length(tw_cca);

rate_bin = [55, 75];
rate_bin_center = [-0.4, 0, 0.4];
nbins = length(rate_bin_center);
cc = cc_ts(cca_tw(tw_cca))';
figure; set(gcf,'color','w');
set(gcf, 'position', [308 14 622 526]);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    subplot(3,1,a); hold on; lstr = {}; x = 0;
    for k = 1:length(cond)
        for tw = 1:length(tw_cca)
            x = x + 1;
            lstr{x} = sprintf('%s-%s', label_str{cond(k)}, ts_str{cca_tw(tw_cca(tw))});
            v = cca_within_auc(idx,cond(k),tw_cca(tw),area_idx);
            vs = cca_within_auc_shuff(idx,cond(k),tw_cca(tw),area_idx,:);
            vs = quantile(vs, 0.95, 5);
            ydata = cell(1,nbins); ym = zeros(1,nbins); yse = zeros(1,nbins);
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i} = v(bin_idx);
                ym(i) = nanmean(ydata{i}); 
                yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
                ysm(i) = nanmean(vs(bin_idx)); 
                ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
                pval = ranksum(ydata{1}, ydata{i});
                plot_pval_star(rate_bin_center(i) + x, ym(i)+yse(i), pval);
            end
            errorbar(rate_bin_center + x, ysm, ysse, 'color', mycc.gray, 'CapSize', 2, 'linewidth', 1);
            errorbar(rate_bin_center + x, ym, yse, 'color', cc{tw}, 'CapSize', 2, 'linewidth', 1);
            ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        end
        if tw==1; ylabel(area_str3{a}); end
        set(gca, 'xtick', []);
    end
end
linkaxes;
% set_figure_style(gcf);


%% print statistics - area pairs
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;
% idx_exp = idx_set & a_idx==0;  tstr = 'S1-RL';
idx_exp = idx_set & a_idx==2;  tstr = 'S1-A';

um = unique(mouse_id(idx_exp));
nsess = zeros(3,length(um));
for n = 1:3
    if n==1; idx = idx_exp & beh_rate<55;
    elseif n==2; idx = idx_exp & beh_rate>=55 & beh_rate<75;
    elseif n==3; idx = idx_exp & beh_rate>=75;
    end
    for i = 1:length(um)
        nsess(n,i) = sum(idx & mouse_id==um(i));
    end
end

%% print statistics - individual areas
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

um = unique(mouse_id(idx_set));
nsess = zeros(3,3,length(um));
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1;
    elseif a==2; idx_exp = idx_set & a_idx==0;
    elseif a==3; idx_exp = idx_set & a_idx==2;
    end
    for n = 1:3
        if n==1; idx = idx_exp & beh_rate<55;
        elseif n==2; idx = idx_exp & beh_rate>=55 & beh_rate<75;
        elseif n==3; idx = idx_exp & beh_rate>=75;
        end
        for i = 1:length(um)
            nsess(a,n,i) = sum(idx & mouse_id==um(i));
        end
    end
end


%% ----------------------- individual variability -----------------------

%% Fig. S6, individual mice within area significant dim, top correlation, and AUC
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

umouse = unique(mouse_id);
N = length(umouse);
cc = {mycc.gray, mycc.pink; mycc.gray, mycc.blue; mycc.gray, mycc.green};

figure; set(gcf, 'color', 'w'); hold on;
set(gcf, 'position', [551 463 906 224]);
nsess = [];
for m = 1:N+1
    for a = 1:3
        if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
        elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
        else; idx_exp = idx_set & a_idx==2; area_idx = 2;
        end
        subplot(3,(N+1),(a-1)*(N+1)+m); hold on; ymi = Inf; yma = -Inf;
        v = cell(1,2);
        for n = 1:2
            if n==1; idx = idx_exp & beh_rate<55;
            else; idx = idx_exp & beh_rate>=75;
            end
            if m<=N; idx = idx & mouse_id==umouse(m); end
            % v{n} = cca_r_within(idx,:,area_idx);  lstr = 'Canonical corr.'; yl = [0.2 0.9];
            % v{n} = ncv_within(idx,:,area_idx);  lstr = 'Significant dim.'; yl = [0 6];
            v{n} = -pred_auc(idx,:,area_idx);  lstr = 'AUC'; yl = [0 1];
            ym = nanmean(v{n}, 1); yse = nanstd(v{n}, [], 1)/sqrt(sum(idx));
            errorbar(1:num_cca, ym, yse, 'color',cc{a,n}, 'CapSize', 4, 'LineWidth', 1);
            nsess(m,n,a) = sum(idx);
        end
        xlim([0.5 num_cca+0.5]);  ylim(yl);
        if m==1; set(gca, 'xtick', []);
        else; set(gca, 'xtick', [], 'ytick', []);
        end
    end
end


%% Fig. S6, individual mice cross area number of significant cvs/top correlation
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

umouse = unique(mouse_id);
N = length(umouse);
cc = {mycc.gray, mycc.blue; mycc.gray, mycc.green};

figure; set(gcf, 'color', 'w'); hold on; mksz = 5; w = 0.2; x_seq = [-0.15 0.15];
set(gcf, 'position', [474 457 889 138])
nsess = [];
for m = 1:N+1
    for a = 1:2
        if a==1; idx_exp = idx_set & a_idx==0; tstr = 'S1-RL';
        else; idx_exp = idx_set & a_idx==2;  tstr = 'S1-A';
        end
        subplot(2,(N+1),(a-1)*(N+1)+m); hold on; ymi = Inf; yma = -Inf; hold on; v = cell(1,2);
        for n = 1:2
            if n==1; idx = idx_exp & beh_rate<55;
            else; idx = idx_exp & beh_rate>=75;
            end
            if m<=N; idx = idx & mouse_id==umouse(m); end
            % v{n} = cca_r_inter(idx,:); lstr = 'Canonical corr.'; yl = [0.2 0.8];
            v{n} = ncv_inter(idx,:);  lstr = 'Significant dim.'; yl = [0 2.5];
            ym = nanmean(v{n}, 1); yse = nanstd(v{n}, [], 1)/sqrt(sum(idx));
            errorbar(1:num_cca, ym, yse,  'color',cc{a,n}, 'CapSize', 4, 'LineWidth', 1);
            nsess(m,n,a) = sum(idx);
        end
        xlim([0.5 num_cca+0.5]);
        if m==1; set(gca, 'xtick', []);
        else; set(gca, 'xtick', [], 'ytick', []);
        end
    end
end
linkaxes; ylim(yl);

%% Fig. S6, within area significant dim, top correlation, and AUC, mouse avg
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

umouse = unique(mouse_id);
N = length(umouse);
cc = {mycc.gray, mycc.pink; mycc.gray, mycc.blue; mycc.gray, mycc.green};

figure; set(gcf, 'color', 'w'); hold on;
set(gcf, 'position', [551 463 906 224]);
nsess = [];
for m = 1:3
    for a = 1:3
        if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
        elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
        else; idx_exp = idx_set & a_idx==2; area_idx = 2;
        end
        subplot(3,3,(m-1)*3+a); hold on; ymi = Inf; yma = -Inf;
        for n = 1:2
            if n==1; idx = idx_exp & beh_rate<55;
            else; idx = idx_exp & beh_rate>=75;
            end
            if m==1
                v0 = cca_r_within(:,:,area_idx);  lstr = 'Canonical corr.'; yl = [0.2 0.9];
            elseif m==2
                v0 = ncv_within(:,:,area_idx);  lstr = 'Significant dim.'; yl = [0 6];
            elseif m==3
                v0 = -pred_auc(:,:,area_idx);  lstr = 'AUC'; yl = [0 1];
            end
            v = cell(N,1);
            valid_idx = true(N,1);
            for i = 1:N
                v{i} = nanmean(v0(idx&mouse_id==umouse(i),:), 1);
                if sum(idx&mouse_id==umouse(i))==0; valid_idx(i) = 0; end
            end
            v = cell2mat(v);
            v = v(valid_idx, :);
            ym = nanmean(v, 1); yse = nanstd(v, [], 1)/sqrt(sum(valid_idx));
            errorbar(1:num_cca, ym, yse, 'color',cc{a,n}, 'CapSize', 4, 'LineWidth', 1);
            nsess(m,n,a) = sum(valid_idx);
        end
        xlim([0.5 num_cca+0.5]);  ylim(yl);
        set(gca, 'xtick', []);
    end
end


%% Fig. S6, cross area number of significant cvs/top correlation, avg by mouse
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

umouse = unique(mouse_id);
N = length(umouse);
cc = {mycc.gray, mycc.blue; mycc.gray, mycc.green};

figure; set(gcf, 'color', 'w'); hold on; mksz = 5; w = 0.2; x_seq = [-0.15 0.15];
set(gcf, 'position', [474 457 889 138])
nsess = [];
for m = 1:2
    for a = 1:2
        if a==1; idx_exp = idx_set & a_idx==0; tstr = 'S1-RL';
        else; idx_exp = idx_set & a_idx==2;  tstr = 'S1-A';
        end
        subplot(2,2,(m-1)*2+a); hold on; ymi = Inf; yma = -Inf; hold on; v = cell(1,2);
        for n = 1:2
            if n==1; idx = idx_exp & beh_rate<55;
            else; idx = idx_exp & beh_rate>=75;
            end
            if m==1
                v0 = cca_r_inter; lstr = 'Canonical corr.'; yl = [0.2 0.8];
            elseif m==2
                v0 = ncv_inter;  lstr = 'Significant dim.'; yl = [0 2.5];
            end

            v = cell(N,1);
            valid_idx = true(N,1);
            for i = 1:N
                v{i} = nanmean(v0(idx&mouse_id==umouse(i),:), 1);
                if sum(idx&mouse_id==umouse(i))==0; valid_idx(i) = 0; end
            end
            v = cell2mat(v);
            v = v(valid_idx, :);

            ym = nanmean(v, 1); yse = nanstd(v, [], 1)/sqrt(sum(valid_idx));
            errorbar(1:num_cca, ym, yse,  'color',cc{a,n}, 'CapSize', 4, 'LineWidth', 1);
            nsess(m,n,a) = sum(valid_idx);
        end
        xlim([0.5 num_cca+0.5]); ylim(yl);
        if a==1; set(gca, 'xtick', []);
        else; set(gca, 'xtick', [], 'ytick', []);
        end
    end
end
ylim(yl);


%% Fig. S6, individual learning curve
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

umouse = unique(mouse_id);
N = length(umouse);

figure; set(gcf, 'color', 'w');
set(gcf, 'position', [474 457 889 138])
for m = 1:N
    subplot(1, N+1, m); hold on;
    idx = idx_set & mouse_id==umouse(m);
	v = beh_rate(idx);
    idx_stop = find(beh_rate(idx)>75, 1);
%     v = v(1:min(idx_stop+3,length(v)));
    % v = smoothdata(v, 'gaussian', 3); 
    plot(v, 'color', 'k', 'linewidth', 1);
    xma = length(v) + 0.1;
    plot([1 xma], 55*[1 1], ':', 'color', mycc.gray);
    plot([1 xma], 75*[1 1], ':', 'color', mycc.gray);
    ylim([30 100]); xlim([1 xma]);
    if m==1; set(gca, 'xtick', []);
    else; set(gca, 'xtick', [], 'ytick', []);
    end
    % title(num2str(length(v)));
    title(num2str(idx_stop));
end