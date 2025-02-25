
%%
addpath('../calcium_analysis');
addpath(genpath('../multiarea_analysis'));
addpath(genpath('./'));

%%
clear; clc
setup_colors;
datasheet = get_data_sheet('multiarea');

%% selected dataset
opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p';

var_to_read = {'num_neuron', 'trial_vec'};
result_name = 'responsive_neuron_learning_no_early_lick_sess_120';

dataset = [131:186,188:256, 389:672]; 

num_ts = 6;
num_trial_type = 14;
tnum = num_trial_type/2;

standard_trial_structure = [0.5,1,1,0.5,2,4];
standard_fr = 10;
standard_ts = make_trial_structure(standard_trial_structure, standard_fr);
standard_ts_fr = cellfun(@(x) x/standard_fr, standard_ts, 'uniformoutput', false);
trial_len = standard_ts{end}(end);
tvec = (1:trial_len)/standard_fr;

%% collect data
num_data = length(dataset);

tw_label = [2,3,4,5,5];
label_str = {'Tone', 'Texture', 'Choice', 'Reward', 'Error'};
ntw = length(label_str);

tw_pair = combvec(1:ntw, 1:ntw);
tw_pair = tw_pair(:,tw_pair(1,:)<tw_pair(2,:));
tw_pair_label = {};
for n = 1:size(tw_pair,2)
    tw_pair_label{end+1} = sprintf('%s-%s', label_str{tw_pair(:,n)});
end

num_responsive = zeros(0,2,ntw);
num_disc = zeros(0,2,ntw,2);
num_responsive_overlap = zeros(0,2,ntw,ntw);
num_disc_overlap = zeros(0,2,ntw,ntw,2);
num_neuron = zeros(0,2);
 
resp_avg = cell(0,2,ntw,tnum);
disc_avg = cell(0,2,ntw,tnum,2); 
disc_joint_avg = cell(0,2,ntw,ntw,tnum,2);

exp_idx = zeros(1,0); 
mouse_id = zeros(1,0); 
a_idx = zeros(1,0); 
beh_rate = zeros(1,0); 
quality_idx = zeros(1,0); 
sess_len = zeros(1,0); 
num_trial = zeros(num_trial_type,0);

count = 0;
for dataid = 1:num_data
    
    dinfo = data_info(datasheet, dataset(dataid), 'multiarea', opts.base_dir);
    if dinfo.quality_idx<2; continue; end
    data = load_data(dinfo, var_to_read, opts);
    spath = fullfile(dinfo.work_dir, opts.result_dir);
    
    % load results
    result_file = fullfile(spath, [result_name '.mat']);
    if ~exist(result_file); continue; end
    ld = load(result_file);
    if length(ld.beh_rate)>2
        [~,s_idx] = max(ld.beh_rate(1:end-1));
        s_idx = s_idx(1);
    else
        s_idx = 1;
    end

    ld.num_sess = 1;
    
    % store experiment information
    eid = get_exp_condition_idx(dinfo);
    exp_idx(count+1:count+ld.num_sess) = eid;
    if strcmp(dinfo.areas, 'A/RL'); a_idx(count+1:count+ld.num_sess) = 1; 
    elseif strcmp(dinfo.areas, 'S1/A'); a_idx(count+1:count+ld.num_sess) = 2;
    else; a_idx(count+1:count+ld.num_sess) = 0;
    end
    mouse_id(count+1:count+ld.num_sess) = dinfo.mouse_id; 
    beh_rate(count+1:count+ld.num_sess) = ld.beh_rate(s_idx);
    quality_idx(count+1:count+ld.num_sess) = dinfo.quality_idx; 
    sess_len(count+1:count+ld.num_sess) = length(ld.sess_idx{s_idx}); 
    
    num_neuron(count+1:count+ld.num_sess,:) = repmat(data.num_neuron,ld.num_sess,1);
    for n = 1:num_trial_type
        num_trial(n,count+1) = sum(data.trial_vec(ld.sess_idx{s_idx})==n);
    end
    
    % cell numbers
    for a = 1:2
        num_responsive(count+1:count+ld.num_sess,a,:) = cellfun(@(x) length(x)/data.num_neuron(a), ...
            ld.responsive_neuron(s_idx,a,:))*100;
        num_disc(count+1:count+ld.num_sess,a,:,:) = cellfun(@(x) length(x)/data.num_neuron(a), ...
            ld.disc_neuron(s_idx,a,:,:))*100;
    end
    
    % overlap between groups
    num_responsive_overlap(count+1:count+ld.num_sess,:,:,:) = ld.num_responsive_overlap(s_idx,:,:,:);
    num_disc_overlap(count+1:count+ld.num_sess,:,:,:,:) = ld.num_disc_overlap(s_idx,:,:,:,:);

    % average traces
    resp_avg(count+1:count+ld.num_sess,:,:,:) = ld.resp_avg(s_idx,:,:,:);
    disc_avg(count+1:count+ld.num_sess,:,:,:,:) = ld.disc_avg(s_idx,:,:,:,:);
    disc_joint_avg(count+1:count+ld.num_sess,:,:,:,:,:) = ld.disc_joint_avg(s_idx,:,:,:,:,:);


    count = count + ld.num_sess;
    
end



%% responsive neurons by performance by area
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
% idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len==120;

rate_bin = [55, 75];  lstr = {'Naive', 'Learning', 'Expert'};
% rate_bin = 50:10:80;
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 4;
set(gcf, 'position', [828 359 720 238]);
for a = 1:3
    if a==1; idx = idx_set; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    else; idx = idx_set & a_idx==2; area_idx = 2;
    end
    subplot(1,3,a); hold on; h = [];
    v_binned = cell(nbins, ntw);
    ym = zeros(nbins,ntw); yse = zeros(nbins,ntw);
    for n = 1:ntw-1
        v = num_responsive(idx,area_idx,n);
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            v_binned{i,n} = v(bin_idx);
            ym(i,n) = nanmean(v(bin_idx)); yse(i,n) = nanstd(v(bin_idx))/sqrt(sum(bin_idx));
        end
%         h(n) = errorbar(rate_bin_center, ym(:,n), yse(:,n), 'Marker','o','MarkerSize',mksz, ...
%             'MarkerFaceColor',cc_ts{n+1}, 'MarkerEdgeColor','none','color',cc_ts{n+1});
        h(n) = errorbar(rate_bin_center, ym(:,n), yse(:,n), 'color',cc_ts{n+1},...
            'capsize', 4, 'linewidth', 1);
        % significance test
        for i = 2:nbins
            pval = ranksum(v_binned{1,n}, v_binned{i,n});
            plot_pval_star(rate_bin_center(i), ym(i,n)+yse(i,n)*1.1, pval);
        end
    end
    xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]); ylim([0 max(ym+yse, [], 'all')+2]);
    title(area_str3{a},'FontWeight','Normal');
    if a==1; ylabel('Percentage'); end
    set(gca, 'xtick', rate_bin_center, 'xticklabel', lstr, 'xticklabelrotation', 45);

end
legend(h, label_str);
% set_figure_style(gcf);
linkaxes;

%% discriminative neurons over learning
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len==120;

tw = 1:ntw-1;
% tw = 4:5;
% rate_bin = 50:10:80;
rate_bin = [55, 75];  lstr = {'Naive', 'Learning', 'Expert'};
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 4;
set(gcf, 'position', [828 359 720 238]);
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    else; idx = idx_set & a_idx==2; area_idx = 2;
    end
    subplot(1,3,a); hold on; h = [];
    v_binned = cell(nbins,length(tw)); 
    ym = zeros(nbins,length(tw)); yse = zeros(nbins,length(tw));
    for n = 1:length(tw)
        v = sum(num_disc(idx,area_idx,tw(n),:),4);
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            v_binned{i,n} = v(bin_idx);
            ym(i,n) = nanmean(v(bin_idx)); yse(i,n) = nanstd(v(bin_idx))/sqrt(sum(bin_idx));
        end
%         h(n) = errorbar(rate_bin_center, ym(:,n), yse(:,n), 'Marker','o','MarkerSize',mksz, ...
%             'MarkerFaceColor',cc_ts{tw(n)+1}, 'MarkerEdgeColor','none','color',cc_ts{tw(n)+1});
        h(n) = errorbar(rate_bin_center, ym(:,n), yse(:,n), 'color',cc_ts{tw(n)+1},...
            'capsize', 4, 'linewidth', 1);
        % significance test
        for i = 2:nbins
            pval = ranksum(v_binned{1,n}, v_binned{i,n});
            plot_pval_star(rate_bin_center(i), ym(i,n)+yse(i,n)*1.1, pval);
        end
    end
    xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]); ylim([0 max(ym+yse, [], 'all')+2]);
    title(area_str3{a},'FontWeight','Normal');
    if a==1; ylabel('Percentage'); end
    set(gca, 'xtick', rate_bin_center, 'xticklabel', lstr, 'xticklabelrotation', 45);
end
legend(h, label_str(tw));
% set_figure_style(gcf);

%% plot number of joint responsive neurons over learning
% idx_set = exp_idx==2;
% idx_set = exp_idx==2 | exp_idx==3 | exp_idx==4;
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len==120;

tw_idx = [1,3,6,  2,4,5];
% tw_idx = 1:size(tw_pair,2);

rate_bin = [55, 75];  lstr = {'Naive', 'Learning', 'Expert'};
% rate_bin = 50:10:80;
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 4;
for n = 1:length(tw_idx)
%     subplot(1,size(tw_pair,2),n); hold on;
    ydata = cell(nbins, 3); ym = zeros(nbins,3); yse = zeros(nbins,3);
    for a = 1:3
        if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
        elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
        else; idx = idx_set & a_idx==2; area_idx = 2;
        end
        subplot(3,length(tw_idx),(a-1)*length(tw_idx)+n); hold on;
%         idx = idx & num_responsive(:,area_idx,tw_pair(1,n))'>5;
%         idx = idx & num_responsive(:,area_idx,tw_pair(2,n))'>5;
        
        v = squeeze(num_responsive_overlap(idx,area_idx,tw_pair(1,tw_idx(n)),tw_pair(2,tw_idx(n)))); 
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            ydata{i,a} = v(bin_idx);
            ym(i,a) = nanmean(v(bin_idx)); yse(i,a) = nanstd(v(bin_idx))/sqrt(sum(bin_idx));
            % test
            try; pval = ranksum(ydata{1,a}, ydata{i,a});
            catch ME; pval = NaN;
            end
            plot_pval_star(rate_bin_center(i), ym(i,a)+yse(i,a), pval);
        end
        errorbar(rate_bin_center, ym(:,a), yse(:,a), 'Marker','o','MarkerSize',mksz, ...
            'MarkerFaceColor',cc_area3{a,1}, 'MarkerEdgeColor','none','color',cc_area3{a,1});
        xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]); ylim([0 max(ym+yse, [], 'all')+2]);
    %     xlabel('Performance');
        if a==1; title(sprintf('%s-\n%s', label_str{tw_pair(:,tw_idx(n))}),'FontWeight','Normal'); end
        if n==1; ylabel(area_str3{a}); end
        if a==3; set(gca, 'xtick', rate_bin_center, 'xticklabel', lstr, 'xticklabelrotation', 45);
        else; set(gca, 'xtick', []);
        end
    end
end
linkaxes;
% legend(h, area_str3);
% set_figure_style(gcf);

%% plot number of joint disc neurons over learning
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

tw_idx = [1,3,6,  2,4,5];
% tw_idx = 1:size(tw_pair,2);

% rate_bin = 50:10:80;
rate_bin = [55, 75];
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 4;
for n = 1:length(tw_idx)
%     subplot(1,size(tw_pair,2),n); hold on;
    ydata = cell(nbins, 3); ym = zeros(nbins,3); yse = zeros(nbins,3);
    for a = 1:3
        if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
        elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
        else; idx = idx_set & a_idx==2; area_idx = 2;
        end
        subplot(3,length(tw_idx),(a-1)*length(tw_idx)+n); hold on;
%         idx = idx & num_responsive(:,area_idx,tw_pair(1,n))'>5;
%         idx = idx & num_responsive(:,area_idx,tw_pair(2,n))'>5;
        
        v = squeeze(num_disc_overlap(idx,area_idx,tw_pair(1,tw_idx(n)),tw_pair(2,tw_idx(n)),1));
%         v = squeeze(num_disc_overlap(idx,area_idx,tw_pair(1,n),tw_pair(2,n),2));  % different preference
        for i = 1:nbins
            if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
            elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
            else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
            end
            ydata{i,a} = v(bin_idx);
            ym(i,a) = nanmean(v(bin_idx)); yse(i,a) = nanstd(v(bin_idx))/sqrt(sum(bin_idx));
        end
        errorbar(rate_bin_center, ym(:,a), yse(:,a), 'Marker','o','MarkerSize',mksz, ...
            'MarkerFaceColor',cc_area3{a,1}, 'MarkerEdgeColor','none','color',cc_area3{a,1});
        % significance test
        for i = 2:nbins
            try; pval = ranksum(ydata{1,a}, ydata{i,a});
            catch ME; pval = NaN;
            end
            plot_pval_star(rate_bin_center(i), max(ym(i,:)+yse(i,:))*1.1, pval);
        end
        xlim([rate_bin_center(1)-10 rate_bin_center(end)+10]); ylim([0 max(ym+yse, [], 'all')+2]);
        if a==1; title(sprintf('%s-\n%s', label_str{tw_pair(:,tw_idx(n))}),'FontWeight','Normal'); end
        if n==1; ylabel(area_str3{a}); end
        if a==3; set(gca, 'xtick', rate_bin_center, 'xticklabel', lstr, 'xticklabelrotation', 45);
        else; set(gca, 'xtick', []);
        end
    end
end
linkaxes;
% legend(h, area_str3);
% set_figure_style(gcf);

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

