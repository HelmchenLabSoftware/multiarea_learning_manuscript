

%%
addpath('../calcium_analysis');
addpath(genpath('../workflow'));
addpath(genpath('../multiarea_analysis'));
addpath(genpath('./'));

%% 
clear;
setup_colors;

% selected dataset
dataset = [132:256, 388:672];
% dataset = 555;  % [608,570,517]
dataset = dataset(end:-1:1);

datasheet = get_data_sheet('multiarea');
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
result_name = 'taskvar_vs_cca_axis_sess_pc30_weighted_tt_50_match';

num_rep = 10; 

result_var = {'beh_rate',  'sess_idx', 'num_sess', 'S_num', ...
    'var_explained', 't_idx_sess', ...
    'ncv_tt_sub', 'cca_r_tt_sub', 'cca_r_tt_sub_shuff', 'ncv_within_tt', ...
    'cca_r_within_tt', 'cca_r_within_tt_shuff', ...
    'svm_cca_sim_tt', 'svm_cca_sim_tt_within', 'cca_sim_tt_sub', ...
    'cca_sim_within_tt', 'cca_sim_inter_vs_within_tt', 'svm_cca_sim_tt_shuff', ...
    'svm_cca_sim_tt_within_shuff', 'cca_sim_tt_sub_shuff', 'cca_sim_within_tt_shuff',...
    'cca_sim_inter_vs_within_tt_shuff', ...
    'cca_within_auc_tt', 'cca_within_auc_tt_shuff', ...
    'cca_inter_auc_tt', 'cca_inter_auc_tt_shuff',...
    'cca_sim_tt_between', 'cca_sim_tt_between_shuff', ...
    'cca_sim_within_tt_between', 'cca_sim_within_tt_between_shuff'};

standard_trial_structure = [0.5,1,1,0.5,2,4];
standard_fr = 10;
standard_ts = make_trial_structure(standard_trial_structure, standard_fr);
standard_ts_fr = cellfun(@(x) x/standard_fr, standard_ts, 'uniformoutput', false);
num_ts = 6;
trial_len = standard_ts{end}(end);
tvec = (1:trial_len)/standard_fr;
num_trial_type = 4;
tnum = num_trial_type/2;

% plot settings
fig_pos_2 = [813 124 564 500];
fig_pos_3 = [359 157 529 683];

%% collect data
num_area = 2;
num_shuff = 100;
num_shuff2 = 50;
% num_shuff2 = 10;
N = num_rep * 2;  % considering two subsets per repeat

nset = length(dataset);
S_num = nan(nset, num_area, N);
var_explained = nan(nset, num_area, N);

% by trial types
ncv_tt_sub = nan(nset, num_cca, tnum, N);
cca_r_tt_sub = nan(nset, num_cca, tnum, N);
ncv_within_tt = nan(nset, num_cca, tnum, num_area, num_rep);
cca_r_within_tt = nan(nset, num_cca, tnum, num_area, num_rep);
cca_r_tt_sub_shuff = nan(nset, num_cca, tnum, num_shuff, N);
cca_r_within_tt_shuff = nan(nset, num_cca, tnum, num_area, num_shuff, num_rep);

cca_sim_tt_inter = nan(nset, num_cca, num_cca, tnum, num_area, N);
cca_sim_within_tt = nan(nset, num_cca, num_cca, tnum, num_area, N);
cca_sim_inter_vs_within_tt = nan(nset, num_cca, num_cca, tnum, num_area, N);
svm_cca_sim_tt = nan(nset, num_cond, num_cca, tnum, num_area, N);
svm_cca_sim_tt_within = nan(nset, num_cond, num_cca, tnum, num_area, N);

cca_sim_tt_inter_shuff = nan(nset, num_cca, num_cca, tnum, num_area, num_shuff, N);
cca_sim_within_tt_shuff = nan(nset, num_cca, num_cca, tnum, num_area, num_shuff, N);
cca_sim_inter_vs_within_tt_shuff = nan(nset, num_cca, num_cca, tnum, num_area, num_shuff, N);
svm_cca_sim_tt_shuff = nan(nset, num_cond, num_cca, tnum, num_area, num_shuff, N);
svm_cca_sim_tt_within_shuff = nan(nset, num_cond, num_cca, tnum, num_area, num_shuff, N);

cca_sim_tt_between = nan(nset, num_cca, num_cca, num_area, N);
cca_sim_tt_between_shuff = nan(nset, num_cca, num_cca, num_area, num_shuff, N);
cca_sim_within_tt_between = nan(nset, num_cca, num_cca, num_area, N);
cca_sim_within_tt_between_shuff = nan(nset, num_cca, num_cca, num_area, num_shuff, N);

cca_within_auc_tt = nan(nset, num_cond, num_cca, tnum, 2, 2);
cca_within_auc_tt_shuff = nan(nset, num_cond, num_cca, tnum, 2, num_shuff2, 2);
cca_inter_auc_tt = nan(nset, num_cond, num_cca, tnum, 2, 2);
cca_inter_auc_tt_shuff = nan(nset, num_cond, num_cca, tnum, 2, num_shuff2, 2);

beh_rate = nan(1,nset); 
exp_idx = nan(1,nset); 
a_idx = zeros(1,nset); 
mouse_id = nan(1,nset); 
quality_idx = nan(1,nset);
sess_len = nan(1,nset);
dataset_id = nan(1,nset);
num_trial = nan(2,nset);

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
    beh_rate(sess_idx) = ld.beh_rate;
    quality_idx(sess_idx) = dinfo.quality_idx;
    dataset_id(sess_idx) = dataset(dataid);
    sess_len(sess_idx) = cellfun(@(x) length(x), ld.sess_idx);
    num_trial(:,sess_idx) = cellfun(@(x) length(x), ld.t_idx_sess)';
    
    %% store all repetition results
    for rep_idx = 1:num_rep
        
        result_file = fullfile(spath, sprintf('%s_rep_%d.mat', result_name, rep_idx));
        if ~exist(result_file); continue; end
        ld = load(result_file, result_var{:});
        
        n_idx = (rep_idx-1)*2+1:rep_idx*2;  % current subsample indices
        sess_idx = count+1:count+ld.num_sess;

        S_num(sess_idx, :, n_idx) = ld.S_num;
        var_explained(sess_idx, :, n_idx) = ld.var_explained;
        
        % by trial type
        ncv_tt_sub(sess_idx, :, :, n_idx) = ld.ncv_tt_sub;
        cca_r_tt_sub(sess_idx, :, :, n_idx) = ld.cca_r_tt_sub;
        cca_r_tt_sub_shuff(sess_idx, :, :, :, n_idx) = permute(ld.cca_r_tt_sub_shuff,[1,2,3,5,4]);
        ncv_within_tt(sess_idx, :, :, :, rep_idx) = ld.ncv_within_tt;
        cca_r_within_tt(sess_idx, :, :, :, rep_idx) = ld.cca_r_within_tt;
        cca_r_within_tt_shuff(sess_idx, :, :, :, :, rep_idx) = ld.cca_r_within_tt_shuff;

        cca_sim_tt_inter(sess_idx, :, :, :, :, n_idx) = ld.cca_sim_tt_sub;
        cca_sim_within_tt(sess_idx, :, :, :, :, n_idx) = ld.cca_sim_within_tt;
        cca_sim_inter_vs_within_tt(sess_idx, :, :, :, :, n_idx) = ld.cca_sim_inter_vs_within_tt;
        svm_cca_sim_tt(sess_idx, :, :, :, :, n_idx) = ld.svm_cca_sim_tt;
        svm_cca_sim_tt_within(sess_idx, :, :, :, :, n_idx) = ld.svm_cca_sim_tt_within;

        cca_sim_tt_inter_shuff(sess_idx, :, :, :, :, :, n_idx) = ld.cca_sim_tt_sub_shuff;
        cca_sim_within_tt_shuff(sess_idx, :, :, :, :, :, n_idx) = ld.cca_sim_within_tt_shuff;
        cca_sim_inter_vs_within_tt_shuff(sess_idx, :, :, :, :, :, n_idx) = ld.cca_sim_inter_vs_within_tt_shuff;
        svm_cca_sim_tt_shuff(sess_idx, :, :, :, :, :, n_idx) = ld.svm_cca_sim_tt_shuff;
        svm_cca_sim_tt_within_shuff(sess_idx, :, :, :, :, :, n_idx) = ld.svm_cca_sim_tt_within_shuff;

        cca_sim_tt_between(sess_idx, :, :, :, n_idx) = ld.cca_sim_tt_between;
        cca_sim_tt_between_shuff(sess_idx, :, :, :, :, n_idx) = ld.cca_sim_tt_between_shuff;
        cca_sim_within_tt_between(sess_idx, :, :, :, n_idx) = ld.cca_sim_within_tt_between;
        cca_sim_within_tt_between_shuff(sess_idx, :, :, :, :, n_idx) = ld.cca_sim_within_tt_between_shuff;

        cca_within_auc_tt(sess_idx, :, :, :, :, n_idx) = ld.cca_within_auc_tt;
        cca_within_auc_tt_shuff(sess_idx, :, :, :, :, :, n_idx) = ld.cca_within_auc_tt_shuff;
        cca_inter_auc_tt(sess_idx, :, :, :, :, n_idx) = ld.cca_inter_auc_tt;
        cca_inter_auc_tt_shuff(sess_idx, :, :, :, :, :, n_idx) = ld.cca_inter_auc_tt_shuff;
    end

    count = count + ld.num_sess;
    
end

%% take absolute value
% by trial type
cca_sim_tt_inter = abs(cca_sim_tt_inter);
cca_sim_within_tt = abs(cca_sim_within_tt);
cca_sim_inter_vs_within_tt = abs(cca_sim_inter_vs_within_tt);
svm_cca_sim_tt = abs(svm_cca_sim_tt);
svm_cca_sim_tt_within = abs(svm_cca_sim_tt_within);

cca_sim_tt_inter_shuff = abs(cca_sim_tt_inter_shuff);
cca_sim_within_tt_shuff = abs(cca_sim_within_tt_shuff);
cca_sim_inter_vs_within_tt_shuff = abs(cca_sim_inter_vs_within_tt_shuff);
svm_cca_sim_tt_shuff = abs(svm_cca_sim_tt_shuff);
svm_cca_sim_tt_within_shuff = abs(svm_cca_sim_tt_within_shuff);

cca_sim_tt_between = abs(cca_sim_tt_between);
cca_sim_tt_between_shuff = abs(cca_sim_tt_between_shuff);
cca_sim_within_tt_between = abs(cca_sim_within_tt_between);
cca_sim_within_tt_between_shuff = abs(cca_sim_within_tt_between_shuff);

% AUC to discrimination index
cca_within_auc_tt = abs(cca_within_auc_tt-0.5) * 2;
cca_within_auc_tt_shuff = abs(cca_within_auc_tt_shuff-0.5) * 2;
cca_inter_auc_tt = abs(cca_inter_auc_tt-0.5) * 2;
cca_inter_auc_tt_shuff = abs(cca_inter_auc_tt_shuff-0.5) * 2;


%% average all repetitions
% by trial type
ncv_tt_sub = nanmean(ncv_tt_sub, 4);
cca_r_tt_sub = nanmean(cca_r_tt_sub, 4);
ncv_within_tt = nanmean(ncv_within_tt, 5);
cca_r_within_tt = nanmean(cca_r_within_tt, 5);
cca_r_tt_sub_shuff = nanmean(cca_r_tt_sub_shuff, 5);
cca_r_within_tt_shuff = nanmean(cca_r_within_tt_shuff, 6);

% average results
cca_sim_tt_inter = nanmean(cca_sim_tt_inter, 6);
cca_sim_within_tt = nanmean(cca_sim_within_tt, 6);
cca_sim_inter_vs_within_tt = nanmean(cca_sim_inter_vs_within_tt, 6);
svm_cca_sim_tt = nanmean(svm_cca_sim_tt, 6);
svm_cca_sim_tt_within = nanmean(svm_cca_sim_tt_within, 6);

% average shuffled results
cca_sim_tt_inter_shuff = nanmean(cca_sim_tt_inter_shuff, 7);
cca_sim_within_tt_shuff = nanmean(cca_sim_within_tt_shuff, 7);
cca_sim_inter_vs_within_tt_shuff = nanmean(cca_sim_inter_vs_within_tt_shuff, 7);
svm_cca_sim_tt_shuff = nanmean(svm_cca_sim_tt_shuff, 7);
svm_cca_sim_tt_within_shuff = nanmean(svm_cca_sim_tt_within_shuff, 7);

cca_sim_tt_between = nanmean(cca_sim_tt_between, 5);
cca_sim_tt_between_shuff = nanmean(cca_sim_tt_between_shuff, 6);
cca_sim_within_tt_between = nanmean(cca_sim_within_tt_between, 5);
cca_sim_within_tt_between_shuff = nanmean(cca_sim_within_tt_between_shuff, 6);

cca_within_auc_tt = nanmean(cca_within_auc_tt, 6);
cca_within_auc_tt_shuff = nanmean(cca_within_auc_tt_shuff, 7);
cca_inter_auc_tt = nanmean(cca_inter_auc_tt, 6);
cca_inter_auc_tt_shuff = nanmean(cca_inter_auc_tt_shuff, 7);


%% ----------------------- trial type analysis ---------------------


%% Fig. S9j-k, plot cross area number of significant cvs/top correlation by trial type, expert only
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & beh_rate>=75;
idx_set = idx_set & num_trial(1,:)>=30;

cc = {mycc.blue; mycc.green};

figure; set(gcf, 'color', 'w'); hold on; mksz = 4; w = 0.2; x_seq = [-0.15 0.15];
% set(gcf, 'position', [655 326 600 256]);
pval_raw = nan(num_cca,2,2);
pval_fdr = nan(num_cca,2,2);
for a = 1:2
    if a==1; idx = idx_set & a_idx==0; tstr = 'S1-RL';
    else; idx = idx_set & a_idx==2;  tstr = 'S1-A';
    end
    for n = 1:2
        subplot(2,2,(n-1)*2+a); hold on; v = cell(1,2); ymi = Inf; yma = -Inf;
        for tt = 1:tnum
            if n==1; v{tt} = cca_r_tt_sub(idx,:,tt); lstr = 'Canonical corr.';
            elseif n==2; v{tt} = ncv_tt_sub(idx,:,tt);  lstr = 'Significant dim.';
            end
            ym = nanmean(v{tt}, 1); yse = nanstd(v{tt}, [], 1)/sqrt(sum(idx));
            h = errorbar(1:num_cca, ym, yse, ...
                'color',cc{a}, 'CapSize', 4, 'LineWidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
        end
        for i = 1:num_cca
            try;pval = signrank(v{1}(:,i), v{tt}(:,i));
            catch ME; pval = NaN;
            end
            pval_raw(i,a,n) = pval;
            % plot_pval_star(i, ym(i)+yse(i), pval_raw(i,a,n));
        end
        pval_fdr(:,a,n) = mafdr(pval_raw(:,a,n));
        p_corrected = pval_raw(:,a,n);
        p_corrected(pval_fdr(:,a,n)>0.05) = 1;
        for i = 1:num_cca
            plot_pval_star(i, ym(i)+yse(i), p_corrected(i));
        end

        xlim([0.5 num_cca+0.5]); 
        set(gca, 'xtick', 1:num_cca, 'xticklabel', ts_str(cca_tw), 'xticklabelrotation', 45);
        ylabel(lstr);
        title(tstr, 'fontweight', 'normal');
    end
end


%% Fig. S9g-h, plot within area number of significant cvs/top correlation by trial type, expert only
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & beh_rate>=75;
idx_set = idx_set & num_trial(1,:)>=30;

figure; set(gcf, 'color', 'w'); hold on; mksz = 4; 
% set(gcf, 'position', [560 326 908 256]);
pval_raw = nan(num_cca,3,2);
pval_fdr = nan(num_cca,3,2);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    else; idx = idx_set & a_idx==2; area_idx = 2;
    end
    for n = 1:2
        subplot(2,3,(n-1)*3+a); hold on; v = cell(1,2);
        for tt = 1:tnum
            if n==1; v{tt} = cca_r_within_tt(idx,:,tt,area_idx);  lstr = 'Canonical corr.';
            elseif n==2; v{tt} = ncv_within_tt(idx,:,tt,area_idx);  lstr = 'Significant dim.';
            end
            ym = nanmean(v{tt}, 1); yse = nanstd(v{tt}, [], 1)/sqrt(sum(idx));
            h = errorbar(1:num_cca, ym, yse, 'color',cc_area3{a,1},  ...
                'CapSize', 4, 'LineWidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
        end
        for i = 1:num_cca
            try; pval = signrank(v{1}(:,i), v{tt}(:,i));
            catch ME; pval = NaN;
            end
            pval_raw(i,a,n) = pval;
        end
        pval_fdr(:,a,n) = mafdr(pval_raw(:,a,n));
        p_corrected = pval_raw(:,a,n);
        p_corrected(pval_fdr(:,a,n)>0.05) = 1;
        for i = 1:num_cca
            plot_pval_star(i, ym(i)+yse(i), p_corrected(i));
        end
        xlim([0.5 num_cca+0.5]); 
        set(gca, 'xtick', 1:num_cca, 'xticklabel', ts_str(cca_tw), 'xticklabelrotation', 45);
        ylabel(lstr); title(area_str3{a}, 'fontweight', 'normal');
    end
end
% set_figure_style(gcf);


%% Fig. 7f, svm vs cca weight similarity matrix, svm only in corresponding window, within model
idx_set = exp_idx==2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & num_trial(1,:)>=30;
idx_set = idx_set & beh_rate>=75;

cond = [1,2,3,4]; tw_cca = 1:4;
% cond = 2; tw_cca = 2;
lstr = {'Correct', 'Incorrect'};
figure; set(gcf,'color','w');
set(gcf, 'position', [418 206 400 638]);
pval_all = nan(length(cond), length(tw_cca), 3);
fdr_all = zeros(size(pval_all));
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    v0 = cell(1,2);
    for tt = 1:2
        subplot(3,2,(a-1)*2+tt); 
        v0{tt} = svm_cca_sim_tt_within(idx,cond,tw_cca,tt,area_idx);
        v = squeeze(nanmean(v0{tt},1));
        vs = svm_cca_sim_tt_within_shuff(idx,cond,tw_cca,tt,area_idx,:);
        vs = quantile(vs, 0.95, 6);
        imagesc(tw_cca, cond, v); hold on;
        % test against shuffled
        sig_map = zeros(length(cond), length(tw_cca));
        for i = 1:length(cond)
            for j = 1:length(tw_cca)
                try; pval = signrank(vs(:,i,j), v0{tt}(:,i,j), 'tail', 'left');
                catch ME; continue;
                end
                if pval<0.05; sig_map(i,j) = 1; end
                if tt==2
                    pval = signrank(v0{1}(:,i,j), v0{tt}(:,i,j));
                    % if pval<0.05; scatter(cond(j), tw_cca(i), 'r*'); end
                    pval_all(i,j,a) = pval;
                end
            end
        end

        % fdr control
        if tt==2
            tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
            fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
            for i = 1:length(cond)
                for j = 1:length(tw_cca)
                    if pval_all(i,j,a)<0.05 && fdr_all(i,j,a)<0.05
                        scatter(cond(j), tw_cca(i), 'r*'); 
                    end
                end
            end
        end

        % overlay significance map
        sc = 100; mgsz = 10;
        sig_lines = find_sig_map_boundary(sig_map, sc, mgsz);
        for i = 1:length(sig_lines)
            plot(sig_lines{i}(:,2)/sc+tw_cca(1)-0.5, sig_lines{i}(:,1)/sc+cond(1)-0.5, 'color', mycc.red);
        end
        clim([0.1 0.4]);
        if a==1; title(lstr{tt}, 'FontWeight', 'Normal'); end
        if tt==1; ylabel(area_str3{a}); end
        if a==3; xlabel('CCA axis'); end
        set(gca, 'xtick', tw_cca, 'xticklabel', ts_str(cca_tw(tw_cca)), 'xticklabelrotation', 45);
        set(gca, 'ytick', cond, 'yticklabel', label_str(cond));
    end
end
colormap(viridis);


%% Fig. 7f, svm vs cca within quantification over learning, correct vs incorrect
idx_set = exp_idx==2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & num_trial(1,:)>=30;

cond = 1:4;
tw = 1:4;

rate_bin = [55, 75];
% rate_bin = 50:10:80;
% rate_bin = 45:10:75;
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
% cc = turbo(length(tw)+1); cc = mat2cell(cc, ones(1,length(tw)+1), 3);

figure; set(gcf,'color','w'); mksz = 4;
set(gcf, 'position', [308 14 622 526]);
pval_all = nan(length(tw), nbins, 3);
fdr_all = nan(length(tw), nbins, 3);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    pval_y = zeros(length(tw), nbins);
    for t = 1:length(tw)
        subplot(3,length(tw),(a-1)*length(tw)+t); hold on;
        ydata = cell(nbins,2); ymi = Inf; yma = -Inf;
        for tt = 1:2
            v = svm_cca_sim_tt_within(idx,cond(t),tw(t),tt,area_idx);
            vs = svm_cca_sim_tt_within_shuff(idx,cond(t),tw(t),tt,area_idx,:);
            vs = quantile(vs, 0.95, 6);
            ym = []; yse = []; ysm = []; ysse = [];
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i,tt} = v(bin_idx);
                ym(i) = nanmean(ydata{i,tt});
                yse(i) = nanstd(ydata{i,tt})/sqrt(sum(bin_idx));
                ysm(i) = nanmean(vs(bin_idx));
                ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
                if tt==2
                    pval = signrank(ydata{i,1}, ydata{i,tt});
                    pval_all(t,i,a) = pval;
                    pval_y(t,i) = ym(i)+yse(i);
                    % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
                end
            end
            h = errorbar(rate_bin_center, ysm, ysse, ...
                'color',mycc.gray, 'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            h = errorbar(rate_bin_center, ym, yse, ...
                'color',cc_ts{cca_tw(tw(t))}, 'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        end
        xlim([rate_bin_center(1)-7 rate_bin_center(end)+7]); ylim([ymi*0.8 yma*1.2]);
        if a==1; title(sprintf('%s\naxis',ts_str{cca_tw(tw(t))}), 'fontweight', 'normal'); end
        if t==1; ylabel(sprintf('%s', area_str3{a})); end
        set(gca, 'xtick', []);
    end

    pval_fdr = mafdr(reshape(pval_all(:,:,a), [], 1));
    pval_fdr = reshape(pval_fdr, size(pval_all,1), size(pval_all,2));
    for t = 1:length(tw)
        subplot(3,length(tw),(a-1)*length(tw)+t); hold on;
        for i = 1:nbins
            if pval_fdr(t,i)<0.05
                plot_pval_star(rate_bin_center(i), pval_y(t,i), pval_all(t,i,a));
            end
        end
    end

end
linkaxes;
% set_figure_style(gcf);
legend('Correct', 'Incorrect');

%% Fig. 7g-h, svm vs cca weight similarity matrix, inter, correct vs incorrect
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & beh_rate>=75;
idx_set = idx_set & num_trial(1,:)>=30;
% idx_set = idx_set & num_trial(1,:)>=20;
% idx_set = idx_set & a_idx==0; area_pair = [1,2];
idx_set = idx_set & a_idx==2; area_pair = [1,3];

cond = [1,2,3,4]; tw_cca = 1:4;
% cond = 2; tw_cca = 3;
lstr = {'Correct', 'Incorrect'};
figure; set(gcf,'color','w');
% set(gcf, 'position', [779 150 432 440]);
pval_all = nan(length(cond), length(tw_cca), 2);
fdr_all = nan(length(cond), length(tw_cca), 2);
for a = 1:2
    v0 = cell(1,2);
    for tt = 1:2
        subplot(2,2,(a-1)*2+tt); 
        v0{tt} = svm_cca_sim_tt(idx_set,cond,tw_cca,tt,a);
        v = squeeze(nanmean(v0{tt},1));
        vs = svm_cca_sim_tt_shuff(idx_set,cond,tw_cca,tt,a,:);
        vs = quantile(vs, 0.95, 6);
        imagesc(tw_cca, cond, v); hold on;

        % statistical test
        sig_map = zeros(length(cond), length(tw_cca));
        for i = 1:length(cond)
            for j = 1:length(tw_cca)
                pval = signrank(vs(:,i,j), v0{tt}(:,i,j), 'tail', 'left');
                if pval<0.05; sig_map(i,j) = 1; end
                if tt==2
                    pval = signrank(v0{1}(:,i,j), v0{tt}(:,i,j));
                    if pval<0.05; scatter(cond(j), tw_cca(i), 'r*'); end
                    pval_all(i,j,a)= pval;
                end
            end
        end

        % fdr control
        if tt==2
            tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
            fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
            for i = 1:length(cond)
                for j = 1:length(tw_cca)
                    if pval_all(i,j,a)<0.05 && fdr_all(i,j,a)<0.05
                        scatter(cond(j), tw_cca(i), 'r*'); 
                    end
                end
            end
        end
        
        % overlay significance map
        sc = 100; mgsz = 10;
        sig_lines = find_sig_map_boundary(sig_map, sc, mgsz);
        for i = 1:length(sig_lines)
            plot(sig_lines{i}(:,2)/sc+tw_cca(1)-0.5, sig_lines{i}(:,1)/sc+cond(1)-0.5, 'color', mycc.red);
        end
        clim([0 0.3]);
        if a==1; title(lstr{tt}, 'FontWeight', 'Normal'); end
        if tt==1; ylabel(area_str3{area_pair(a)}); end
        if a==2; xlabel('CCA axis'); end
        set(gca, 'xtick', tw_cca, 'xticklabel', ts_str(cca_tw(tw_cca)), 'xticklabelrotation', 45);
        set(gca, 'ytick', cond, 'yticklabel', label_str(cond));
    end
end
% set_figure_style(gcf);
colormap(viridis);




%% Fig. 7g-h, svm vs cca inter quantification over learning, correct vs incorrect
idx = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx = idx & num_trial(1,:)>=30;
idx = idx & a_idx==0; area_pair = [1,2];
% idx = idx & a_idx==2; area_pair = [1,3];

cond = 1:4; tw_cca = 1:4;
% cond = 2; tw_cca = 2;

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
cc = turbo(length(tw_cca)+1); cc = mat2cell(cc, ones(1,length(tw_cca)+1), 3);

figure; set(gcf,'color','w'); mksz = 4;
set(gcf, 'position', [593 121 622 313]);
pval_all = nan(length(tw_cca), nbins, 2);
pval_shuff = nan(length(tw_cca), nbins, 2, 2);
for a = 1:2
    pval_y = zeros(length(tw_cca), nbins);
    for t = 1:length(tw_cca)
        subplot(2,length(tw_cca),(a-1)*length(tw_cca)+t); hold on;
        ydata = cell(nbins,2); ymi = Inf; yma = -Inf;
        for tt = 1:2
            v = svm_cca_sim_tt(idx,cond(t),tw_cca(t),tt,a);
            vs = svm_cca_sim_tt_shuff(idx,cond(t),tw_cca(t),tt,a,:);
            ym = []; yse = []; ysm = []; ysse = [];
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i,tt} = v(bin_idx);
                ym(i) = nanmean(ydata{i,tt});
                yse(i) = nanstd(ydata{i,tt})/sqrt(sum(bin_idx));
                ysm(i) = nanmean(vs(bin_idx));
                ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
                pval_shuff(t,i,a,tt) = signrank(vs(bin_idx), v(bin_idx), 'tail', 'left');
                if tt==2
                    pval = signrank(ydata{i,1}, ydata{i,tt});
                    pval_all(t,i,a) = pval;
                    pval_y(t,i) = ym(i)+yse(i);
                    % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
                end
            end
            h = errorbar(rate_bin_center, ysm, ysse,...
                'color', mycc.gray, 'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            h = errorbar(rate_bin_center, ym, yse,...
                'color',cc_ts{cca_tw(tw_cca(t))}, 'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        end
        xlim([rate_bin_center(1)-7 rate_bin_center(end)+7]); ylim([ymi*0.8 yma*1.2]);
        if a==1; title(sprintf('%s axis',ts_str{cca_tw(tw_cca(t))}), 'fontweight', 'normal'); end
        if t==1; ylabel(sprintf('%s', area_str3{area_pair(a)})); end
        set(gca, 'xtick', []);
    end

    pval_fdr = mafdr(reshape(pval_all(:,:,a), [], 1));
    pval_fdr = reshape(pval_fdr, size(pval_all,1), size(pval_all,2));
    for t = 1:length(tw_cca)
        subplot(2,length(tw_cca),(a-1)*length(tw_cca)+t); hold on;
        for i = 1:nbins
            if pval_fdr(t,i)<0.05 && any(pval_shuff(t,i,:)<0.05)
                plot_pval_star(rate_bin_center(i), pval_y(t,i), pval_all(t,i,a));
            end
        end
    end

end
linkaxes;
legend('Correct', 'Incorrect');


%% Fig. S10a, within axis projection AUC quantification
idx_set = exp_idx==2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & num_trial(1,:)>=30;

cond = 1:3;
tw = 1:3;

rate_bin = [55, 75];
% rate_bin = 50:10:80;
% rate_bin = 45:10:75;
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
% cc = turbo(length(tw)+1); cc = mat2cell(cc, ones(1,length(tw)+1), 3);

figure; set(gcf,'color','w'); mksz = 4;
set(gcf, 'position', [308 14 622 526]);
pval_all = nan(length(tw),nbins,3);
pval_shuff = nan(length(tw),nbins,3,2);
pval_y = nan(length(tw),nbins,3);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    for t = 1:length(tw)
        subplot(3,length(tw),(a-1)*length(tw)+t); hold on;
        ydata = cell(nbins,2); ymi = Inf; yma = -Inf;
        for tt = 1:2
            v = cca_within_auc_tt(idx,cond(t),tw(t),tt,area_idx);
%             v = abs(v);
            vs = cca_within_auc_tt_shuff(idx,cond(t),tw(t),tt,area_idx,:);
            vs = quantile(vs, 0.95, 6);
            ym = []; yse = []; ysm = []; ysse = [];
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i,tt} = v(bin_idx);
                ym(i) = nanmean(ydata{i,tt});
                yse(i) = nanstd(ydata{i,tt})/sqrt(sum(bin_idx));
                ysm(i) = nanmean(vs(bin_idx));
                ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
                pval_shuff(t,i,a,tt) = signrank(vs(bin_idx), v(bin_idx), 'tail', 'left');
                if tt==2
                    pval = signrank(ydata{i,1}, ydata{i,tt});
                    pval_all(t,i,a) = pval;
                    pval_y(t,i,a) = ym(i)+yse(i);
                    % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
                end
            end
            h = errorbar(rate_bin_center, ysm, ysse, ...
                'color', mycc.gray, 'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            h = errorbar(rate_bin_center, ym, yse, ...
                'color',cc_ts{cca_tw(tw(t))}, 'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        end
        xlim([rate_bin_center(1)-7 rate_bin_center(end)+7]); ylim([ymi*0.8 yma*1.2]);
        if a==1; title(sprintf('%s\naxis',ts_str{cca_tw(tw(t))}), 'fontweight', 'normal'); end
        if t==1; ylabel(sprintf('%s', area_str3{a})); end
        set(gca, 'xtick', []);
    end
end
linkaxes;
legend('Correct', 'Incorrect');

fdr_all = zeros(size(pval_all));
for a = 1:3
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    for t = 1:length(tw)
        subplot(3,length(tw),(a-1)*length(tw)+t); hold on;
        for i = 1:nbins
            if fdr_all(t,i,a)<0.05 && any(pval_shuff(t,i,a,:)<0.05)
                plot_pval_star(rate_bin_center(i), pval_y(t,i,a), pval_all(t,i,a));
            end
        end
    end
end

%% Fig. S10b-c, inter axis projection AUC quantification
idx = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx = idx & quality_idx==2;
idx = idx & num_trial(1,:)>=30;
idx = idx & a_idx==0; area_pair = [1,2];
% idx = idx & a_idx==2; area_pair = [1,3];

tw = 1:3; cond = 1:3;
% tw = 2; cond = 2;

rate_bin = [55, 75];
% rate_bin = 50:10:80;
% rate_bin = 45:10:75;
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
cc = turbo(length(tw)+1); cc = mat2cell(cc, ones(1,length(tw)+1), 3);

figure; set(gcf,'color','w'); mksz = 4;
set(gcf, 'position', [593 121 622 313]);
pval_all = nan(length(tw),nbins,2);
pval_shuff = nan(length(tw),nbins,2,2);
pval_y = nan(length(tw),nbins,2);
for a = 1:2
    for t = 1:length(tw)
        subplot(2,length(tw),(a-1)*length(tw)+t); hold on;
        ydata = cell(nbins,2); ymi = Inf; yma = -Inf;
        for tt = 1:2
            v = cca_inter_auc_tt(idx,cond(t),tw(t),tt,a);
            vs = cca_inter_auc_tt_shuff(idx,cond(t),tw(t),tt,a,:);
            vs = quantile(vs, 0.95, 6);
            ym = []; yse = [];
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i,tt} = v(bin_idx);
                ym(i) = nanmean(ydata{i,tt});
                yse(i) = nanstd(ydata{i,tt})/sqrt(sum(bin_idx));
                ysm(i) = nanmean(vs(bin_idx));
                ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
                pval_shuff(t,i,a,tt) = signrank(vs(bin_idx), v(bin_idx), 'tail', 'left');
                if tt==2
                    pval = signrank(ydata{i,1}, ydata{i,tt});
                    pval_all(t,i,a) = pval;
                    pval_y(t,i,a) = ym(i)+yse(i);
                    % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);   
                end
            end
            h = errorbar(rate_bin_center, ysm, ysse, ...
                'color', mycc.gray, 'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            h = errorbar(rate_bin_center, ym, yse, ...
                'color',cc_ts{cca_tw(tw(t))}, 'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        end
        xlim([rate_bin_center(1)-7 rate_bin_center(end)+7]); ylim([ymi*0.8 yma*1.2]);
        if a==1; title(sprintf('%s axis',ts_str{cca_tw(tw(t))}), 'fontweight', 'normal'); end
        if t==1; ylabel(sprintf('%s', area_str3{area_pair(a)})); end
        set(gca, 'xtick', []);
    end
end
linkaxes;
% set_figure_style(gcf);
legend('Correct', 'Incorrect');

fdr_all = zeros(size(pval_all));
for a = 1:2
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    for t = 1:length(tw)
        % tmp = mafdr(reshape(pval_all(t,:,a), [], 1));
        % fdr_all(t,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
        subplot(2,length(tw),(a-1)*length(tw)+t); hold on;
        for i = 1:nbins
            if fdr_all(t,i,a)<0.05 && any(pval_shuff(t,i,a,:)<0.05)
                plot_pval_star(rate_bin_center(i), pval_y(t,i,a), pval_all(t,i,a));
            end
        end
    end
end

%% print statistics
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_exp = idx_set & a_idx==0;  tstr = 'S1-RL';
% idx_exp = idx_set & a_idx==2;  tstr = 'S1-A';


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

