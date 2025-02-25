
%%
clear; clc
setup_colors;
datasheet = get_data_sheet('multiarea');

%% selected dataset
opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p';

var_to_read = {'rate', 'num_neuron', 'trial_vec'};
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

num_responsive = zeros(num_data,2,ntw);
num_disc = zeros(num_data,2,ntw,2);
num_responsive_overlap = zeros(num_data,2,ntw,ntw);
num_disc_overlap = zeros(num_data,2,ntw,ntw,2);
num_neuron = zeros(num_data,2);
 
resp_avg = cell(num_data,2,ntw,tnum);
disc_avg = cell(num_data,2,ntw,tnum,2); 
disc_joint_avg = cell(num_data,2,ntw,ntw,tnum,2);

exp_idx = zeros(1,num_data); 
mouse_id = zeros(1,num_data); 
a_idx = zeros(1,num_data); 
beh_rate = zeros(1,num_data); 
quality_idx = zeros(1,num_data); 
sess_len = zeros(1,num_data); 
num_trial = zeros(num_trial_type,num_data);

count = 0;
for dataid = 1:num_data
    
    dinfo = data_info(datasheet, dataset(dataid), 'multiarea', opts.base_dir);
    if dinfo.quality_idx==0; continue; end
    data = load_data(dinfo, var_to_read, opts);
    spath = fullfile(dinfo.work_dir, opts.result_dir);
    
    % load results
    ld = load(fullfile(spath, [result_name '.mat']));
    ld.num_sess = length(ld.sess_idx);
    
    % store experiment information
    eid = get_exp_condition_idx(dinfo);
    exp_idx(count+1:count+ld.num_sess) = eid;
    if strcmp(dinfo.areas, 'A/RL'); a_idx(count+1:count+ld.num_sess) = 1; 
    elseif strcmp(dinfo.areas, 'S1/A'); a_idx(count+1:count+ld.num_sess) = 2;
    else; a_idx(count+1:count+ld.num_sess) = 0;
    end
    mouse_id(count+1:count+ld.num_sess) = dinfo.mouse_id; 
    beh_rate(count+1:count+ld.num_sess) = ld.beh_rate;
    quality_idx(count+1:count+ld.num_sess) = dinfo.quality_idx; 
    sess_len(count+1:count+ld.num_sess) = cellfun(@(x) length(x), ld.sess_idx); 
    
    num_neuron(count+1:count+ld.num_sess,:) = repmat(data.num_neuron,ld.num_sess,1);
    for m = 1:ld.num_sess
        for n = 1:num_trial_type
            num_trial(n,count+m) = sum(data.trial_vec(ld.sess_idx{m})==n);
        end
    end
    
    % cell numbers
    for a = 1:2
        num_responsive(count+1:count+ld.num_sess,a,:) = cellfun(@(x) length(x)/data.num_neuron(a), ...
            ld.responsive_neuron(:,a,:))*100;
        num_disc(count+1:count+ld.num_sess,a,:,:) = cellfun(@(x) length(x)/data.num_neuron(a), ...
            ld.disc_neuron(:,a,:,:))*100;
    end
    
    % overlap between groups
    num_responsive_overlap(count+1:count+ld.num_sess,:,:,:) = ld.num_responsive_overlap;
    num_disc_overlap(count+1:count+ld.num_sess,:,:,:,:) = ld.num_disc_overlap;

    % average traces
    resp_avg(count+1:count+ld.num_sess,:,:,:) = ld.resp_avg;
    disc_avg(count+1:count+ld.num_sess,:,:,:,:) = ld.disc_avg;
    disc_joint_avg(count+1:count+ld.num_sess,:,:,:,:,:) = ld.disc_joint_avg;

    count = count + ld.num_sess;
    
end

%% number of imaged neurons each area, three areas together
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;
idx_set = idx_set & all(num_neuron'>30);

figure; set(gcf,'color','w'); hold on;
w = 0.5; mksz = 10; r = 0.7; yma = -Inf;
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    nx = sum(idx);
    v = num_neuron(idx,area_idx); v = v(:);
    scatter(a*ones(nx,1)+w*(rand(nx,1)-0.5), v, mksz,...
        cc_area3{a,2}, 'filled', 'markerfacealpha', r);
    h = boxplot(v, 'position', a, 'width', w, 'colors', cc_area3{a,1});
    yl = setBoxStyle(h, 1);
    yma = max(yma, yl(2)+5);
    if a==1; v0 = v; end
    pval = ranksum(v0, v);
    disp(pval);
end
xlim([0.5 3.5]); ylim([0 yma]);
set(gca,'xtick',1:3,'xticklabel',area_str3);
ylabel('Number of neurons'); box off
set_figure_style(gcf);

%% number of imaged neurons each area, naive vs expert
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;
idx_set = idx_set & all(num_neuron'>30);

num_neuron_expert = cell(3,1);
figure; set(gcf,'color','w'); hold on; x_seq = [-0.15, 0.15];
w = 0.25; mksz = 10; r = 0.7; yma = -Inf; h = [];
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx_exp = idx_set & a_idx==2; area_idx = 2;
    end
    v = cell(1,2); ys = 0;
    for n = 1:2
        if n==1; idx = idx_exp & beh_rate<55; cc = mycc.gray;
        elseif n==2; idx = idx_exp & beh_rate>=75; cc = cc_area3{a,1};
        end
        nx = sum(idx);
        v{n} = num_neuron(idx,area_idx); v{n} = v{n}(:);
        % scatter((a+x_seq(n))*ones(nx,1)+w*(rand(nx,1)-0.5), v, mksz,...
        %     cc, 'filled', 'markerfacealpha', r);
        h(:,n) = boxplot(v{n}, 'position', a+x_seq(n), 'width', w, 'colors', cc);
        yl = setBoxStyle(h(:,n), 1);
        ys = max(ys, yl(2));
        yma = max(yma, yl(2)+5);
        pval = ranksum(v{1}, v{n});
        plot_pval_star(a, ys*1.1, pval);
        % disp(pval);
        if n==2; num_neuron_expert{a} = v{n}; end
    end
end
xlim([0.5 3.5]); ylim([0 yma*1.1]);
set(gca,'xtick',1:3,'xticklabel',area_str3);
ylabel('Number of neurons'); box off
set_figure_style(gcf);
legend(h(1,:), 'Naive', 'Expert')

%% responsive neurons by performance by area
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

rate_bin = [55, 75];  lstr = {'Naive', 'Learning', 'Expert'};
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 4;
set(gcf, 'position', [828 359 720 238]);
pval_all = nan(ntw-1, nbins-1, 3);
fdr_all = nan(size(pval_all));
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
        h(n) = errorbar(rate_bin_center, ym(:,n), yse(:,n), 'color',cc_ts{n+1},...
            'capsize', 4, 'linewidth', 1);
        % significance test
        for i = 2:nbins
            pval = ranksum(v_binned{1,n}, v_binned{i,n});
            pval_all(n,i-1,a) = pval;
            % plot_pval_star(rate_bin_center(i), ym(i,n)+yse(i,n)*1.1, pval);
        end
    end
    % multiple comparison correction
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    for n = 1:ntw-1
        for i = 2:nbins
            if fdr_all(n,i-1,a)<0.05
                plot_pval_star(rate_bin_center(i), ym(i,n)+yse(i,n)*1.1, pval_all(n,i-1,a));
            end
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
idx_set = idx_set & sess_len>=80;

tw = 1:ntw-1;
rate_bin = [55, 75];  lstr = {'Naive', 'Learning', 'Expert'};
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 4;
set(gcf, 'position', [828 359 720 238]);
pval_all = nan(ntw-1, nbins-1, 3);
fdr_all = nan(size(pval_all));
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
        h(n) = errorbar(rate_bin_center, ym(:,n), yse(:,n), 'color',cc_ts{tw(n)+1},...
            'capsize', 4, 'linewidth', 1);
        % significance test
        for i = 2:nbins
            pval = ranksum(v_binned{1,n}, v_binned{i,n});
            pval_all(n,i-1,a) = pval;
            % plot_pval_star(rate_bin_center(i), ym(i,n)+yse(i,n)*1.1, pval);
        end
    end

    % multiple comparison correction
    try; tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    catch ME; tmp = zeros(size(pval_all,1), size(pval_all,2));
    end
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    for n = 1:ntw-1
        for i = 2:nbins
            if fdr_all(n,i-1,a)<0.05
                plot_pval_star(rate_bin_center(i), ym(i,n)+yse(i,n)*1.1, pval_all(n,i-1,a));
            end
        end
    end

    xlim([rate_bin_center(1)-5 rate_bin_center(end)+5]); ylim([0 max(ym+yse, [], 'all')+2]);
    title(area_str3{a},'FontWeight','Normal');
    if a==1; ylabel('Percentage'); end
    set(gca, 'xtick', rate_bin_center, 'xticklabel', lstr, 'xticklabelrotation', 45);
end
legend(h, label_str(tw));

%% plot number of joint responsive neurons over learning
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

tw_idx = [1,3,6,  2,4,5];

rate_bin = [55, 75];  lstr = {'Naive', 'Learning', 'Expert'};
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);
figure; set(gcf,'color','w'); mksz = 4;
pval_all = nan(length(tw_idx), nbins, 3);
for n = 1:length(tw_idx)
    ydata = cell(nbins, 3); ym = zeros(nbins,3); yse = zeros(nbins,3);
    for a = 1:3
        if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
        elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
        else; idx = idx_set & a_idx==2; area_idx = 2;
        end
        subplot(3,length(tw_idx),(a-1)*length(tw_idx)+n); hold on;
        
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
            pval_all(n,i,a) = pval;
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
pval_all = nan(length(tw_idx), nbins-1, 3);
for n = 1:length(tw_idx)
%     subplot(1,size(tw_pair,2),n); hold on;
    ydata = cell(nbins, 3); ym = zeros(nbins,3); yse = zeros(nbins,3);
    for a = 1:3
        if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
        elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
        else; idx = idx_set & a_idx==2; area_idx = 2;
        end
        subplot(3,length(tw_idx),(a-1)*length(tw_idx)+n); hold on;
        
        v = squeeze(num_disc_overlap(idx,area_idx,tw_pair(1,tw_idx(n)),tw_pair(2,tw_idx(n)),1));
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
            pval_all(n,i-1,a) = pval;
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

% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:3
    tmp = mafdr(reshape(pval_all(:,:,a), [], 1));
    fdr_all(:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2));
    for n = 1:length(tw_idx)
        subplot(3,length(tw_idx),(a-1)*length(tw_idx)+n); hold on;
        for i = 2:nbins
            if fdr_all(n,i-1,a)<0.05 && pval_all(n,i-1,a)<0.05
                scatter(rate_bin_center(i), 0, 'ro');
            end
        end    
    end
end


%% disc neuron avg naive vs expert
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;
% idx_set = idx_set & beh_rate<55;
% idx_set = idx_set & beh_rate>75;

trial = 1;

tw = 1:4;
figure; set(gcf,'color','w');
% set(gcf,'position',[580 388 830 510]);
for a = 1:3
    if a==1; idx_exp = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx_exp = idx_set & a_idx==0; area_idx = 2;
    else; idx_exp = idx_set & a_idx==2; area_idx = 2;
    end
    for n = 1:length(tw)
        cc_learning = {mycc.gray, cc_ts{tw(n)+1}};
        subplot(3,length(tw),(a-1)*length(tw)+n); hold on; h = [];
        for m = 1:2
            if m==1; idx = idx_exp & beh_rate<55;
            elseif m==2; idx = idx_exp & beh_rate>75;
            end
%             for i = 1:2
            for i = 1
                v = disc_avg(idx,area_idx,tw(n),trial,i);
                v = cell2mat(v);
                ym = nanmean(v);  yse = nanstd(v)/sqrt(size(v,1));
                h(end+1,:) = confplot(tvec, ym, yse, yse, cc_learning{m}, 0.2);
                if i==2; set(h(end,1), 'linestyle', '--'); end
            end
        end
        ylim([-0.3 1.7]);
        if n==1; ylim([-0.3 2.2]); end
        draw_trial_structure(standard_ts_fr(2:5));
        xlim([0.1 6.5]); 
        if a==1; title(sprintf('%s\nneurons', label_str{tw(n)}), 'FontWeight', 'Normal'); end
        if n==1; ylabel(area_str3{a}); end
        if a==3; xlabel('Time (s)'); end
    end
end
legend(h(:,1), {'Preferred, Naive', 'Nonpreferred, Naive', ...
    'Preferred, Expert', 'Nonpreferred, Expert'});
% set_figure_style(gcf);

%% disc neuron response across windows over learning
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

trial = 1;
tw_neuron = 1:4;
tw_data = 2:5;

rate_bin = [55, 75];  xstr = {'Naive', 'Learning', 'Expert'};
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

pval_all = zeros(length(tw_neuron), length(tw_data), nbins-1, 3);
pval_y = zeros(length(tw_neuron), length(tw_data), nbins-1, 3);
figure; set(gcf,'color','w'); mksz = 4; h = [];
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    for n = 1:length(tw_neuron)
        subplot(3,length(tw_neuron),(a-1)*length(tw_neuron)+n); hold on; 
        yma = 0; ymi = 0; lstr = {};
        v0 = disc_avg(idx,area_idx,tw_neuron(n),trial,1);
        v0 = cell2mat(v0);
        for m = 1:length(tw_data)
            v = nanmean(v0(:,standard_ts{tw_data(m)}),2);
            ydata = cell(1,nbins);
            ym = zeros(1,nbins); yse = zeros(1,nbins);
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{i} = v(bin_idx);
                ym(i) = nanmean(ydata{i}); yse(i) = nanstd(ydata{i})/sqrt(sum(bin_idx));
                % test
                if i>1
                    pval = ranksum(ydata{1}, ydata{i});
                    % pval = pval * (nbins-1);
                    pval_all(n,m,i-1,a) = pval;
                    pval_y(n,m,i-1,a) = ym(i)+yse(i);
                    % plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
                end
            end
            % plot
            cc = cc_ts{tw_data(m)};
            if m~=n; cc = lighten_color(cc, 0.5); end
            h(m) = errorbar(rate_bin_center, ym, yse, 'color',cc, 'linewidth', 1, 'capsize', 2);
            ymi = min(ymi, min(ym-yse)); yma = max(yma, max(ym+yse));
            lstr{m} = [ts_str{tw_data(m)} ' window'];
        end
        xlim([rate_bin_center(1)-10 rate_bin_center(end)+10]); 
        ylim([ymi*1.2 yma*1.2]);
%         set(gca,'ytick', [0 1]);
        if a==1; title(sprintf('%s\nneurons', label_str{tw_neuron(n)}), 'FontWeight', 'Normal'); end
        if n==1; ylabel(area_str3{a}); end
        if a==3
            set(gca,'xtick',rate_bin_center,'xticklabel',xstr,'xticklabelrotation',45); 
        else
            set(gca,'xtick',[]);
        end
    end
end
legend(h, lstr);
linkaxes


% fdr control
fdr_all = zeros(size(pval_all));
for a = 1:3
    tmp = mafdr(reshape(pval_all(:,:,:,a), [], 1));
    fdr_all(:,:,:,a) = reshape(tmp, size(pval_all,1), size(pval_all, 2), size(pval_all, 3));
    for n = 1:length(tw_neuron)
        subplot(3,length(tw_neuron),(a-1)*length(tw_neuron)+n); hold on; 
        for m = 1:length(tw_data)
            for i = 2:nbins
                if fdr_all(n,m,i-1,a)<0.05
                    plot_pval_star(rate_bin_center(i), pval_y(n,m,i-1,a), pval_all(n,m,i-1,a));
                end
            end
        end
    end
end

%% -------------- correct vs incorrect ---------------
%% disc neuron avg traces in expert, correct vs incorrecy
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & beh_rate>=75;
idx_set = idx_set & sess_len>=80;

trial = [1,2];
cc = {mycc.gray, mycc.orange};
i = 1; % preferred variable

figure; set(gcf,'color','w');
% set(gcf,'position',[680 709 1120 389])
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    else; idx = idx_set & a_idx==2; area_idx = 2;
    end
    for n = 1:ntw
        subplot(3,ntw,(a-1)*ntw+n); hold on; h = [];
        for t = 1:length(trial)
            v = disc_avg(idx,area_idx,n,trial(t),i);
            v = cell2mat(v);
            ym = nanmean(v);  yse = nanstd(v)/sqrt(size(v,1));
            h = confplot(tvec, ym, yse, yse, cc_ts{tw_label(n)}, 0.2);
            if t==2; set(h(1), 'linestyle', '--'); end
        end
        draw_trial_structure(standard_ts_fr);
        xlim([0 7]);
        if a==1; title(label_str{n}, 'FontWeight', 'Normal'); end
        if n==1; ylabel(lstr{m}); end
    end
end
% set_figure_style(gcf);

%% disc neuron response avg, correct vs incorrect, quantification
idx_set = exp_idx<=2 | ((exp_idx==3 | exp_idx==4) & beh_rate>70);
idx_set = idx_set & quality_idx==2;
idx_set = idx_set & sess_len>=80;

trial = [1,2];
tw_neuron = 1:4;

rate_bin = [55, 75];  xstr = {'Naive', 'Learning', 'Expert'};
binsz = diff(rate_bin);
rate_bin_center = [rate_bin(1)-mean(binsz)/2, rate_bin(1:end-1)+binsz/2, rate_bin(end)+mean(binsz)/2];
nbins = length(rate_bin_center);

figure; set(gcf,'color','w'); mksz = 4; h = [];
for a = 1:3
    if a==1; idx = idx_set & a_idx~=1; area_idx = 1;
    elseif a==2; idx = idx_set & a_idx==0; area_idx = 2;
    elseif a==3; idx = idx_set & a_idx==2; area_idx = 2;
    end
    for n = 1:length(tw_neuron)
        subplot(3,length(tw_neuron),(a-1)*length(tw_neuron)+n); hold on; 
        yma = 0; ymi = 0; lstr = {};
        ydata = cell(length(trial),nbins);
        for t = 1:length(trial)
            v0 = disc_avg(idx,area_idx,tw_neuron(n),trial(t),1);
            v0 = cell2mat(v0);
            v = nanmean(v0(:,standard_ts{tw_label(tw_neuron(n))}),2);
            ym = zeros(1,nbins); yse = zeros(1,nbins);
            for i = 1:nbins
                if i==1; bin_idx = beh_rate(idx)<rate_bin(1);
                elseif i==nbins; bin_idx = beh_rate(idx)>=rate_bin(end);
                else; bin_idx = beh_rate(idx)>=rate_bin(i-1) & beh_rate(idx)<rate_bin(i);
                end
                ydata{t,i} = v(bin_idx);
                ym(i) = nanmean(ydata{t,i}); yse(i) = nanstd(ydata{t,i})/sqrt(sum(bin_idx));
                % test
                pval = signrank(ydata{1,i}, ydata{t,i});
                pval = pval * nbins;
                plot_pval_star(rate_bin_center(i), ym(i)+yse(i), pval);
            end
            % plot
            cc = cc_ts{tw_label(tw_neuron(n))};
            h = errorbar(rate_bin_center, ym, yse, 'color',cc, 'linewidth', 1, 'capsize', 4);
            if t==2; set(h, 'linestyle', '--'); end
            ymi = min(ymi, min(ym-yse)); yma = max(yma, max(ym+yse));
        end
        xlim([rate_bin_center(1)-10 rate_bin_center(end)+10]); 
        ylim([ymi*1.2 yma*1.2]);
        if a==1; title(sprintf('%s\nneurons', label_str{tw_neuron(n)}), 'FontWeight', 'Normal'); end
        if n==1; ylabel(area_str3{a}); end
        if a==3
            set(gca,'xtick',rate_bin_center,'xticklabel',xstr,'xticklabelrotation',45); 
        else
            set(gca,'xtick',[]);
        end
    end
end
% set_figure_style(gcf);
linkaxes
