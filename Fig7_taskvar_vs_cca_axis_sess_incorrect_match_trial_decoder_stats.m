

%%
addpath('../calcium_analysis');
addpath('../multiarea_analysis');
addpath(genpath('./'));

%% 
clear; clc
setup_colors;

% selected dataset
dataset = [132:256, 388:672];
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
result_name = 'taskvar_vs_cca_axis_sess_pc30_tt_50_match_decoder_auc';

num_rep = 10; 

result_var = {'beh_rate',  'sess_idx', 'num_sess', 'S_num', ...
    'var_explained', 't_idx_sess', 'pred_auc', 'pred_auc_shuff', 'svm_sim'};


num_trial_type = 4;
tnum = num_trial_type/2;

% plot settings
fig_pos_2 = [813 124 564 500];
fig_pos_3 = [359 157 529 683];

%% collect data
num_area = 2;
num_shuff = 100;
N = num_rep * 2;  % considering two subsets per repeat

nset = length(dataset);
S_num = nan(nset, num_area, N);
var_explained = nan(nset, num_area, N);

% by trial types
pred_auc = nan(nset, num_cond, 2, tnum, N);
pred_auc_shuff = nan(nset, num_cond, 2, tnum, num_shuff, N);
svm_sim = nan(nset, num_cond, num_cond, 2, tnum, N);

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
        
        S_num(sess_idx, :, n_idx) = ld.S_num;
        var_explained(sess_idx, :, n_idx) = ld.var_explained;
        
        % by trial type
        pred_auc(sess_idx, :, :, :, n_idx) = ld.pred_auc;
        pred_auc_shuff(sess_idx, :, :, :, :, n_idx) = ld.pred_auc_shuff;
        
        svm_sim(sess_idx, :, :, :, :, n_idx) = ld.svm_sim;
        
    end

    count = count + ld.num_sess;
    
end


%% convert to d prime
% AUC to discrimination index
pred_auc = (pred_auc-0.5) * 2;
pred_auc_shuff = (pred_auc_shuff-0.5) * 2;

svm_sim = abs(svm_sim);

%% average all repetitions
pred_auc = nanmean(pred_auc, 5);
pred_auc_shuff = nanmean(pred_auc_shuff, 6);

svm_sim = nanmean(svm_sim, 6);

%% Fig. 7c, auc three areas over learning
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & num_trial(1,:)>=30;

cond = [1,2,3];

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

lstr = {'Naive', 'Learning', 'Expert'};
figure; set(gcf,'color','w'); mksz = 4;
pval_all = nan(length(cond), nbins, 3);
pval_y = nan(length(cond), nbins, 3);
for k = 1:length(cond)
    for a = 1:3
        if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
        elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
        elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
        end
        subplot(3,length(cond),(a-1)*length(cond)+k); hold on; ymi = Inf; yma = -Inf;
        ydata = cell(nbins, tnum);
        ym = nan(1,nbins); yse = nan(1,nbins); 
        ysm = nan(1,nbins); ysse = nan(1,nbins); 
        for tt = 1:tnum
            v = pred_auc(idx,cond(k),tt,area_idx);
            vs = squeeze(pred_auc_shuff(idx,cond(k),tt,area_idx,:));
            vs = quantile(vs, 0.95, 2);
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ysm(i) = nanmean(vs(bin_idx)); ysse(i) = nanstd(vs(bin_idx))/sqrt(sum(bin_idx));
                ydata{i,tt} = v(bin_idx);
                ym(i) = nanmean(ydata{i,tt}); yse(i) = nanstd(ydata{i,tt})/sqrt(sum(bin_idx));
                % test between trial types
                try; pval = signrank(ydata{i,1}, ydata{i,tt});
                catch ME; pval = NaN;
                end
                pval_all(k,i,a) = pval;
                pval_y(k,i,a) = (ym(i)+yse(i))*1.05;
                % plot_pval_star(rate_bin_center(i), (ym(i)+yse(i))*1.05, pval);
            end
            h = errorbar(rate_bin_center, ysm, ysse, 'color', mycc.gray, ...
                'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            h = errorbar(rate_bin_center, ym, yse, 'color', cc_ts{cond_tw(cond(k))}, ...
                'CapSize', 4, 'linewidth', 1);
            if tt==2; set(h, 'linestyle', '--'); end
            ymi = nanmin([ymi,ysm-ysse]); yma = nanmax([yma,ysm+ysse]);
            ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        end
        xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]); ylim([ymi yma]); box off
        ylim([-0.55 1]);
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
        for i = 1:nbins
            if fdr_all(k,i-1,a)<0.05
                plot_pval_star(rate_bin_center(i), pval_y(k,i-1,a), pval_all(k,i-1,a));
            end
        end
    end
end
