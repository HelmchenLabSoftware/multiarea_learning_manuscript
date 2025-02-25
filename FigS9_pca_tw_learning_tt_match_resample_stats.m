clear; clc
setup_colors;
datasheet = get_data_sheet('multiarea');
 
%% selected dataset
dataset = [131:256, 388:473, 475:672];
% dataset = [529:672];
 
opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p'; 
var_to_read = {'trial_vec', 'num_neuron'};
% result_name = 'pca_tw_learning_tt_match_resample_50';  % 20 reps
result_name = 'pca_tw_learning_tt_match_resample_50_rep50';  % 50 reps

% num_rep = 20;
num_rep = 50;
num_pc = 5;
num_cond = 4;
num_ts = 6;
tnum = 2;

%% initialize results
nset = length(dataset);

num_pc_neuron_tt = nan(0, num_ts, tnum, 2);
num_pc_time_tt = nan(0, num_ts, tnum, 2);
pc_auc_tt = nan(0, num_cond, tnum, 2, num_pc, num_rep);

num_neuron = nan(0, 2);
exp_idx = zeros(1,nset); 
beh_rate = nan(1,nset); 
a_idx = zeros(1,nset); 
mouse_id = zeros(1,nset);
quality_idx = zeros(1,nset);
dataset_id = zeros(1,nset);
num_trial = zeros(tnum,nset);
sess_len = zeros(1,nset);
 
% collect data 
count = 0;
for dataid = 1:nset
    
    %% dataset information
    dinfo = data_info(datasheet, dataset(dataid), 'multiarea', opts.base_dir);
    spath = fullfile(dinfo.work_dir, opts.result_dir);
    data = load_data(dinfo, var_to_read, opts);
    result_file = fullfile(spath, [result_name '.mat']);
    if ~exist(result_file); continue; end

    ld = load(result_file);
    
    % store experiment information
    eid = get_exp_condition_idx(dinfo);
    exp_idx(count+1:count+ld.num_sess) = eid;
    if strcmp(dinfo.areas, 'A/RL'); a_idx(count+1:count+ld.num_sess) = 1; 
    elseif strcmp(dinfo.areas, 'S1/A'); a_idx(count+1:count+ld.num_sess) = 2;
    else; a_idx(count+1:count+ld.num_sess) = 0;
    end
    s_idx = count+1:count+ld.num_sess;
    mouse_id(s_idx) = dinfo.mouse_id; 
    beh_rate(s_idx) = ld.beh_rate;
    quality_idx(s_idx) = dinfo.quality_idx; 
    dataset_id(s_idx) = dataset(dataid); 
    sess_len(s_idx) = cellfun(@(x) length(x), ld.sess_idx); 
    num_neuron(s_idx,:) = ld.num_neuron;
    num_trial(:,s_idx) = cellfun(@(x) length(x), ld.t_idx)';
    
    %% store data
    num_pc_neuron_tt(count+1:count+ld.num_sess,:,:,:) = nanmean(ld.num_pc_neuron_tt(:,:,:,:,1:num_rep), 5);
    num_pc_time_tt(count+1:count+ld.num_sess,:,:,:) = nanmean(ld.num_pc_time_tt(:,:,:,:,1:num_rep), 5);
    
    pc_auc_tt(count+1:count+ld.num_sess,:,:,:,:,:) = ld.pc_auc_tt(:,:,:,:,:,1:num_rep);

    count = count + ld.num_sess;

end

%% Fig. S9d, plot timewise pc by trial type, only expert
idx_set = exp_idx==2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & num_trial(1,:)>=30;
idx_set = idx_set & beh_rate>=75; 

tw = 2:5;
% tw = 1:6;
figure; set(gcf, 'color', 'w'); hold on; h = [];
pval_all = nan(length(tw), 3);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2;  area_idx = 2;
    end
    subplot(1,3,a); hold on; cc = cc_area3{a,1}; v = cell(1,2);
    for tt = 1:tnum
        % v{tt} = num_pc_neuron_tt(idx,tw,tt,area_idx);
        v{tt} = num_pc_time_tt(idx,tw,tt,area_idx);
        ym = nanmean(v{tt}, 1); yse = nanstd(v{tt}, [], 1)/sqrt(sum(idx));
        h = errorbar(tw, ym, yse, 'color', cc, 'CapSize', 4, 'LineWidth', 1);
        if tt==2; set(h, 'linestyle', '--'); end
        for i = 1:length(tw)
            pval = signrank(v{1}(:,i), v{tt}(:,i));
            pval_all(i,a) = pval;
            plot_pval_star(tw(i), ym(i)+yse(i), pval);
        end
    end
    
    xlim([tw(1)-0.5 tw(end)+0.5]);
    title(area_str3{a}, 'FontWeight', 'Normal');
    xlabel('Time (s)');  
    if a==1; ylabel('Percentage of PC'); end
end
% set_figure_style(gcf);

fdr_all = nan(size(pval_all));
for a = 1:3
    fdr_all(:,a) = mafdr(pval_all(:,a));
end
% fdr_all = mafdr(reshape(pval_all, [], 1));
% fdr_all = reshape(fdr_all, size(pval_all,1), size(pval_all, 2));

%% Fig. S9e, AVG top PC AUC over learning in corresponding window by trial type
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>=70);
idx_set = idx_set & all(num_neuron'>30);
idx_set = idx_set & num_trial(1,:)>=30;

k_set = [1,2,3];

trial = [1,2];
rate_bin = [55 75];
% rate_bin = 50:10:80;
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 3;
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2; 
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    subplot(3,1,a); hold on;
    
    v0 = permute(pc_auc_tt(idx,:,:,area_idx,:,:), [1,2,3,5,6,4]);  % all PCs
    % v0 = permute(pc_auc_tt(idx,:,:,area_idx,1,:), [1,2,3,5,6,4]);  % first PC
    v0(v0==0) = NaN;
    v0 = (v0-0.5)*2;
    
    for k = 1:length(k_set)
        
        h = []; ymi = Inf; yma = -Inf; 
        cc = cc_ts{tw(k)};
        ydata = cell(nbins,2);
        
        % determine the sign of d' according to correct trial
        for n = 1:size(v0, 4)
            for i = 1:size(v0,1)
                for j = 1:num_rep
                    if nanmean(v0(i,:,1,n,j))<0  % average across all variables
                        v0(i,:,1,n,j) = -v0(i,:,1,n,j);
                        v0(i,:,2,n,j) = -v0(i,:,2,n,j);
                    end
                end
            end
        end
        
        for t = 1:length(trial)
            v = nanmean(nanmean(v0(:,k_set(k),trial(t),:,:), 5), 4);
            ym = []; yse = [];
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i,t} = v(bin_idx);
                ym(i) = nanmean(ydata{i,t});
                yse(i) = nanstd(ydata{i,t})/sqrt(sum(bin_idx));
                pval = signrank(ydata{i,1}, ydata{i,t});
%                     pval = pval * nbins;
                plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
            end
            h = errorbar(rate_bin_center, ym, yse,  ...
                'color',cc, 'CapSize', 4, 'linewidth', 1);
            if t==2; set(h, 'linestyle', '--'); end
            ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        end
        
        xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]);
%         ylim([ymi*0.8 yma*1.2]);
        if a==1; title('Disc. Index', 'FontWeight', 'Normal'); end
        if k==1; ylabel(area_str3{a}); end
        if a==3
            set(gca, 'xtick', rate_bin_center, 'xticklabelrotation', 45, ...
                'xticklabel', {'Naive', 'Learning', 'Expert'});
        else
            set(gca, 'xtick', []);
        end
    end
end
linkaxes;