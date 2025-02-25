addpath(genpath('../multiarea_analysis'))

clear; clc
setup_colors;
datasheet = get_data_sheet('multiarea');
 
%% selected dataset
dataset = [131:256, 388:473, 475:672];

opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p'; 
var_to_read = {'trial_vec', 'num_neuron'};
result_name = 'pca_timewise_learning_sess_120_resampled';

standard_trial_structure = [0.5,1,1,0.5,2,4];
standard_fr = 10;
standard_ts = make_trial_structure(standard_trial_structure, standard_fr);
standard_ts_fr = cellfun(@(x) x/standard_fr, standard_ts, 'uniformoutput', false);
trial_len = standard_ts{end}(end);
tvec = (1:trial_len)/standard_fr;
 
num_rep = 10;
% num_rep = 50;
num_pc = 5;
num_cond = 4;

num_ts = 6;
tnum = 2;

%% initialize results
nset = length(dataset);

num_pc_neuron = nan(0, trial_len, 2, num_rep);
num_pc_time = nan(0, trial_len, 2, num_rep);

pc_auc = nan(0, trial_len, num_cond, 2, num_pc, num_rep);
num_neuron = nan(0,  2, num_rep);

exp_idx = zeros(1,nset); 
beh_rate = nan(1,nset); 
a_idx = zeros(1,nset); 
mouse_id = zeros(1,nset);
quality_idx = zeros(1,nset);
sess_len = zeros(1,nset);

% collect data 
count = 0;
for dataid = 1:nset
    
    %% dataset information
    dinfo = data_info(datasheet, dataset(dataid), 'multiarea', opts.base_dir);
    if dinfo.quality_idx==0; continue; end
    eid = get_exp_condition_idx(dinfo);
    if eid>4; continue; end

    spath = fullfile(dinfo.work_dir, opts.result_dir);
    data = load_data(dinfo, var_to_read, opts);

    result_file = fullfile(spath, sprintf('%s_rep_%d.mat', result_name, 1));
    ld = load(result_file);
    
    % store experiment information
    exp_idx(count+1:count+ld.num_sess) = eid;
    if strcmp(dinfo.areas, 'A/RL'); a_idx(count+1:count+ld.num_sess) = 1; 
    elseif strcmp(dinfo.areas, 'S1/A'); a_idx(count+1:count+ld.num_sess) = 2;
    else; a_idx(count+1:count+ld.num_sess) = 0;
    end
    mouse_id(count+1:count+ld.num_sess) = dinfo.mouse_id; 
    beh_rate(count+1:count+ld.num_sess) = ld.beh_rate;
    quality_idx(count+1:count+ld.num_sess) = dinfo.quality_idx; 
    sess_len(count+1:count+ld.num_sess) = cellfun(@(x) length(x), ld.sess_idx); 

    for rep_idx = 1:num_rep
        % result_file = fullfile(spath, [result_name '.mat']);
        result_file = fullfile(spath, sprintf('%s_rep_%d.mat', result_name, rep_idx));
        if ~exist(result_file); continue; end
    
        ld = load(result_file);

        % store data
        s_idx = count+1:count+ld.num_sess;
        num_pc_neuron(s_idx,:,:,rep_idx) = ld.num_pc_neuron;
        num_pc_time(s_idx,:,:,rep_idx) = ld.num_pc_time;
        pc_auc(s_idx,:,:,:,:,rep_idx) = ld.pc_auc;
        num_neuron(s_idx,:,rep_idx) = ld.num_neuron;
    end

    count = count + ld.num_sess;

end

num_pc_time = num_pc_time*100;

% average all repeats
num_pc_neuron = nanmean(num_pc_neuron, 4);
num_pc_time = nanmean(num_pc_time, 4);
pc_auc = abs(pc_auc-0.5)*2; 
pc_auc = nanmean(pc_auc, 6);
num_neuron = nanmean(num_neuron, 3);

%% plot number of neurons
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;
idx_set = idx_set & all(num_neuron'>30);

num_neuron_expert = cell(3,1);
figure; set(gcf,'color','w'); hold on; x_seq = [-0.15, 0.15];
w = 0.25; mksz = 10; r = 0.7; yma = -Inf;
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx_exp = idx_set & a_idx==2; area_idx = 2;
    end
    for n = 1:2
        if n==1; idx = idx_exp & beh_rate<55;
        elseif n==2; idx = idx_exp & beh_rate>=75;
        end
        nx = sum(idx);
        v = num_neuron(idx,area_idx); v = v(:);
        scatter((a+x_seq(n))*ones(nx,1)+w*(rand(nx,1)-0.5), v, mksz,...
            cc_area3{a,2}, 'filled', 'markerfacealpha', r);
        h = boxplot(v, 'position', a+x_seq(n), 'width', w, 'colors', cc_area3{a,1});
        yl = setBoxStyle(h, 1);
        yma = max(yma, yl(2)+5);
        if n==1; v0 = v; end
        pval = ranksum(v0, v);
        plot_pval_star(a, yl(2)*1.1, pval);
        % disp(pval);
        if n==2; num_neuron_expert{a} = v; end
    end
end
xlim([0.5 3.5]); ylim([0 yma]);
set(gca,'xtick',1:3,'xticklabel',area_str3);
ylabel('Number of neurons'); box off
set_figure_style(gcf);

%% plot timewise pc naive vs expert
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
idx_set = idx_set & all(num_neuron'>30);

smooth_wsz = 3;
t_idx = standard_ts{1}(1):standard_ts{5}(end)+20;
figure; set(gcf, 'color', 'w'); hold on; h = [];
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx_exp = idx_set & a_idx==2;  area_idx = 2;
    end
    subplot(1,3,a); hold on; 
    for i = 1:2
        if i==1; idx = idx_exp & beh_rate<55; cc = mycc.gray;
        else; idx = idx_exp & beh_rate>=75; cc = cc_area3{a,1};
        end
        % v = num_pc_neuron(idx,:,area_idx);
        v = num_pc_time(idx,:,area_idx);
        v = smoothdata(v,2,'gaussian',smooth_wsz);
        ym = nanmean(v, 1); yse = nanstd(v, [], 1)/sqrt(sum(idx));
        h(a,:) = confplot(tvec(t_idx), ym(t_idx), yse(t_idx), yse(t_idx), cc, 0.2);
    end
    xlim([tvec(t_idx(1)) tvec(t_idx(end))]);
    title(area_str3{a}, 'FontWeight', 'Normal');
    xlabel('Time (s)');  
    if a==1; ylabel('Percentage of PC'); end
end
linkaxes;
xlim([0.1 6.2]);
for a = 1:3
    subplot(1,3,a); draw_trial_structure(standard_ts_fr(2:5));
end


%% population timewise pc over learning combined plot
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
idx_set = idx_set & all(num_neuron'>30);

tw = 2:5;
rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 4;
pval_all = zeros(length(tw), nbins-1, 3);
pval_y = zeros(length(tw), nbins-1, 3);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2;  area_idx = 2;
    end
    subplot(1, 3, a); hold on; h = []; ymi = Inf; yma = -Inf; 
    for n = 1:length(tw)
        ydata = cell(1,nbins);
        % v = num_pc_neuron(idx,standard_ts{tw(n)},area_idx);
        v = num_pc_time(idx,standard_ts{tw(n)},area_idx);
        v = nanmean(v,2);
        ym = []; yse = [];
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            ydata{i} = v(bin_idx);
            ym(i) = nanmean(v(bin_idx));
            yse(i) = nanstd(v(bin_idx))/sqrt(sum(bin_idx));
        end
        h(end+1) = errorbar(rate_bin_center, ym, yse, ...
            'color',cc_ts{tw(n)}, 'CapSize', 3, 'LineWidth', 1);
        for i = 2:nbins
            try; pval = ranksum(ydata{1}, ydata{i});
            catch ME; pval = NaN;
            end
            % pval = pval * (nbins-1);
            pval_all(n,i-1,a) = pval;
            pval_y(n,i-1,a) = ym(i)+yse(i);
            % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
        end
        ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
    end
    xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]);
    ylim([ymi*0.8 yma*1.2]);
    set(gca, 'xtick', rate_bin_center, 'xticklabelrotation', 45, ...
        'xticklabel', {'Naive', 'Learning', 'Expert'});
    title(area_str3{a}, 'FontWeight', 'Normal');
end
% set_figure_style(gcf);
legend(h, ts_str(tw));
linkaxes;

% fdr control
% fdr_all = zeros(size(pval_all));
% for a = 1:3
%     tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
%     fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
%     subplot(1, 3, a); hold on;
%     for n = 1:length(tw)
%         for i = 2:nbins
%             if fdr_all(n,i-1,a)<0.05
%                 plot_pval_star(rate_bin_center(i), pval_y(n,i-1,a), pval_all(n,i-1,a));
%             end
%         end
%     end
% end

% fdr control
fdr_all = zeros(size(pval_all));
tmp = mafdr(reshape(pval_all, [], 1));
fdr_all = reshape(tmp, size(pval_all,1), size(pval_all, 2), size(pval_all, 3));
for a = 1:3
    subplot(1, 3, a); hold on;
    for n = 1:length(tw)
        for i = 2:nbins
            if fdr_all(n,i-1,a)<0.05
                plot_pval_star(rate_bin_center(i), pval_y(n,i-1,a), pval_all(n,i-1,a));
            end
        end
    end
end

%% top PC auc
idx_set = exp_idx==2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
idx_set = idx_set & all(num_neuron'>30);

k_set = [1,1,2,3];
tw = [2,3,4,5];

rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 4;
pval_all = zeros(length(k_set), num_pc, 2, 3);
pval_y = zeros(length(k_set), num_pc, 2, 3);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2; 
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    for k = 1:length(k_set)
        subplot(3,length(k_set),(a-1)*length(k_set)+k); hold on; 
        h = []; ymi = Inf; yma = -Inf; ydata = cell(1,nbins);
        cc = cc_ts{tw(k)};
        cc = {darken_color(cc,0.8), darken_color(cc,0.4), cc, ...
            lighten_color(cc,0.4), lighten_color(cc,0.8)};
        for n = 1:num_pc
            v = pc_auc(idx,:,k_set(k),area_idx,n);
            v(v==0) = NaN;
            ym = []; yse = [];
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i} = nanmean(v(bin_idx,standard_ts{tw(k)}),2);
                ym(i) = nanmean(ydata{i});
                yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
            end
            h(end+1) = errorbar(rate_bin_center, ym, yse, ...
                'color',cc{n}, 'CapSize', 3, 'LineWidth', 1);
            for i = 2:nbins
                try; pval = ranksum(ydata{1}, ydata{i});
                catch ME; pval = NaN;
                end
                % pval = pval * (nbins-1);
                pval_all(k,n,i-1,a) = pval;
                pval_y(k,n,i-1,a) = ym(i)+yse(i);
                % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
            end
            ymi = nanmin([ymi,ym-yse]); yma = nanmax([yma,ym+yse]);
        end
        xlim([rate_bin_center(1)-3 rate_bin_center(end)+3]);
        ylim([ymi*0.8 yma*1.2]);
        if a==1; title(sprintf('%s',ts_str{tw(k)}), 'FontWeight', 'Normal'); end
        if k==1; ylabel(sprintf('%s\nDisc. Index', area_str3{a})); end
        if a==3
            set(gca, 'xtick', rate_bin_center, 'xticklabelrotation', 45, ...
                'xticklabel', {'Naive', 'Learning', 'Expert'});
        else
            set(gca, 'xtick', []);
        end
    end
end
% set_figure_style(gcf);
linkaxes;

% fdr control
fdr_all = zeros(size(pval_all));
tmp = mafdr(reshape(pval_all, [], 1));
fdr_all = reshape(tmp, size(pval_all,1), size(pval_all, 2), size(pval_all, 3), 3);
for a = 1:3
    for k = 1:length(k_set)
        subplot(3,length(k_set),(a-1)*length(k_set)+k); hold on; 
        for n = 1:num_pc
            for i = 2:nbins
                if fdr_all(k,n,i-1,a)<0.05
                    plot_pval_star(rate_bin_center(i), pval_y(k,n,i-1,a), pval_all(k,n,i-1,a));
                end
            end
        end
    end
end

%% stats
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

trial_stats = zeros(3,nbins);
rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2;  area_idx = 2;
    end
    for i = 1:nbins
        if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
        elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
        else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
        end
        trial_stats(a,i) = sum(bin_idx);
    end
end

