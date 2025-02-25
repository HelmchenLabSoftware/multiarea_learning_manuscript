
clear all; clc;
setup_colors;
datasheet = get_data_sheet('multiarea');

%%
opts = struct;
opts.base_dir = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\multiarea';
opts.data_dir = 'data_suite2p';
opts.result_dir = 'results_suite2p';
% result_name = 'lick_behavior_sess';
result_name = 'lick_behavior_sess_2';

dataset = [131:256, 389:672];

trial_str_combined = ystr{1};
num_trial_type = 4;
tnum = num_trial_type/2;
choice_type = get_trial_type_choice(1:num_trial_type);

% standard trial structure and frame rate
standard_trial_structure = [0.5,1,1,2,2,4];
standard_fr = 10;
standard_ts = make_trial_structure(standard_trial_structure, standard_fr);
standard_ts_fr = cellfun(@(x) x/standard_fr, standard_ts, 'uniformoutput', false);
trial_len = standard_ts{end}(end);

%% pull results together
beh_rate = nan(0, 1);
lick_time = cell(0,tnum);
lick_cue = cell(0,2,tnum);
lick_cue_perc = zeros(0,2,tnum);
lick_tex = cell(0,2,tnum);
lick_rate = cell(0,tnum);
standard_resample_ts = cell2mat(standard_ts_fr(2:4)');
standard_resample_ts = standard_resample_ts - standard_resample_ts(1);
prev_rate = zeros(0,tnum,3);
next_rate = zeros(0,tnum,3);
mouse_id = [];
exp_idx = [];
data_idx = [];
sess_len = [];
count = 0;

for dataid = dataset
    
    dinfo = data_info(datasheet, dataid, 'multiarea', opts.base_dir);
    spath = fullfile(dinfo.work_dir, opts.result_dir);
    result_file = fullfile(spath, [result_name '.mat']);
    if ~exist(result_file); continue; end

    ld = load(result_file);
    nsess = length(ld.sess_idx);
    beh_rate(count+1:count+nsess,:) = ld.beh_rate_combined(:,:,1);
    data = load_data(dinfo, {'ts', 'trial_structure'}, opts);

    % lick rate before texture onset
    for m = 1:nsess
        for n = 1:tnum
            % response time
            lick_time{count+m,n} = cat(1, ld.lick_first{n*2-1}, ld.lick_first{n*2}) / ...
                data.trial_structure(2)*standard_trial_structure(2);
    
            % lick on final choice port
            if choice_type(n*2-1)~=0
                v = cat(1,ld.lick_rate.cue{m,choice_type(n*2-1),n*2-1},...
                    ld.lick_rate.cue{m,choice_type(n*2),n*2});
            else
                v = cat(1,ld.lick_rate.cue{m,1,n*2-1},ld.lick_rate.cue{m,2,n*2});
            end
            lick_cue{count+m,1,n} = v;
            lick_cue_perc(count+m,1,n) = sum(v~=0)/length(v);
            % incorrect pre lick
            if choice_type(n*2-1)~=0
                v = cat(1,ld.lick_rate.cue{m,setdiff(1:2,choice_type(n*2-1)),n*2-1},...
                    ld.lick_rate.cue{m,setdiff(1:2,choice_type(n*2)),n*2});
            else
                v = cat(1,ld.lick_rate.cue{m,2,n*2-1},ld.lick_rate.cue{m,1,n*2});
            end
            lick_cue{count+m,2,n} = v;
            lick_cue_perc(count+m,2,n) = sum(v~=0)/length(v);
        end
        
        % lick rate during texture
        for n = 1:tnum
            % lick on final choice port
            if choice_type(n*2-1)~=0
                v = cat(1,ld.lick_rate.texture{m,choice_type(n*2-1),n*2-1},...
                    ld.lick_rate.texture{m,choice_type(n*2),n*2});
            else
                v = cat(1,ld.lick_rate.texture{m,1,n*2-1},ld.lick_rate.texture{m,2,n*2});
            end
            lick_tex{count+m,1,n} = v;
            % incorrect lick
            if choice_type(n*2-1)~=0
                v = cat(1,ld.lick_rate.texture{m,setdiff(1:2,choice_type(n*2-1)),n*2-1},...
                    ld.lick_rate.texture{m,setdiff(1:2,choice_type(n*2)),n*2});
            else
                v = cat(1,ld.lick_rate.texture{m,2,n*2-1},ld.lick_rate.texture{m,1,n*2});
            end
            lick_tex{count+m,2,n} = v;
        end
    end
    
    % lick rate over trial time
    ts0 = cell(1,2);
    idx = find(ld.resample_ts>data.trial_structure(2),1); 
    ts0{1} = 1:idx;
    idx = find(ld.resample_ts>sum(data.trial_structure(2:3)),1); 
    if isempty(idx); idx = length(ld.resample_ts); end
    ts0{2} = ts0{1}(end):idx;
    idx = find(ld.resample_ts>sum(data.trial_structure(2:4)),1); 
    if isempty(idx); idx = length(ld.resample_ts); end
    ts0{3} = ts0{2}(end):idx;
    ts1 = cellfun(@(x) x-standard_ts{1}(end), standard_ts(2:4), 'uniformoutput', 0);
    ld.lick_port_rate = cellfun(@(x) align_trial_to_standard_2d(x, ts0, ts1), ld.lick_port_rate,...
        'uniformoutput', 0);
    for m = 1:nsess
        for n = 1:tnum
            lick_rate{count+m,n,1} = cat(1, ld.lick_port_rate{m,2*n-1,1}, ld.lick_port_rate{m,2*n,2});
            lick_rate{count+m,n,2} = cat(1, ld.lick_port_rate{m,2*n,1}, ld.lick_port_rate{m,2*n-1,2});
        end
    end
    
    mouse_id(count+1:count+nsess) = dinfo.mouse_id;
    data_idx(count+1:count+nsess) = dataid;
    exp_idx(count+1:count+nsess) = get_exp_condition_idx(dinfo);
    sess_len(count+1:count+nsess) = cellfun(@(x) length(x), ld.sess_idx);
    
    count = count + nsess;
    
end

beh_rate = beh_rate' * 100;

%% lick rate over time
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

trial = [1,2]; 
port_idx = [1,2]; % 1, correct port; 2, incorrect port
tex_onset = standard_ts_fr{3}(1) - standard_ts_fr{2}(1);
report_onset = standard_ts_fr{4}(1) - standard_ts_fr{2}(1);

cc = {mycc.gray, mycc.orange};
figure; set(gcf,'color','w'); hold on;
for t = 1:length(trial)
    subplot(length(trial),1,t); hold on;
    for n = 1:2
        if n==1; idx = idx_set & beh_rate<55;
        elseif n==2; idx = idx_set & beh_rate>75;
        end
        v = lick_rate(idx,trial(t),port_idx(t));
        v = cellfun(@(x) movmean(x, [1,5], 2, 'omitnan'), v, 'uniformoutput', 0);
        v = cellfun(@(x) nanmean(x,1), v, 'uniformoutput', 0);
        v = cell2mat(v);
        ym = nanmean(v,1)*100; yse = nanstd(v,[],1)/sqrt(size(v,1))*100;
        confplot(standard_resample_ts, ym, yse, yse, cc{n}, 0.2);
    end
    yl = get(gca, 'ylim');
    plot(tex_onset*[1 1], yl, 'k:');
    plot(report_onset*[1 1], yl, 'k:');
    xlim([0 2.5]); ylim(yl); 
    xlabel('Time (s)'); ylabel('Percentage of trials');
    title(trial_str_combined{trial(t)}, 'FontWeight', 'Normal');
end
% set_figure_style(gcf);

%% quantify lick rate
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;

trial = [1,2]; 
% port_idx = [1,1]; % 1, correct port; 2, incorrect port
port_idx = [1,2]; % 1, correct port; 2, incorrect port

plot_type = 'violin';
lstr = {'Naive', 'Learning', 'Expert'};
cc = {mycc.gray, (mycc.gray+mycc.orange)/2, mycc.orange};

figure; set(gcf,'color','w'); hold on; r = 0.7;
set(gcf, 'position', [1014 354 180 337]);
for t = 1:length(trial)
    subplot(length(trial),1,t); hold on; yma = 0; ymi = Inf;
    for n = 1:3
        if n==1; idx = idx_set & beh_rate<55;
        elseif n==2; idx = idx_set & beh_rate>55 & beh_rate<75;
        elseif n==3; idx = idx_set & beh_rate>75;
        end

        % % tone window
        v = lick_rate(idx,trial(t),port_idx(t));
        v = cellfun(@(x) nanmean(x(:,standard_ts{2}-standard_ts{1}(end))*standard_fr,'all'), v);
        
        % first 0.5s of texture
        % v = lick_rate(idx,trial(t),port_idx(t));
        % v = cellfun(@(x) nanmean(x(:,standard_ts{3}(1:5)-standard_ts{1}(end))*standard_fr,'all'), v);
        
        % last 0.5s of texture
%         v = lick_rate(idx,trial(t),port_idx(t));
%         v = cellfun(@(x) nanmean(x(:,standard_ts{3}(end-4:end)-standard_ts{1}(end))*standard_fr,'all'), v);
        
        v = v(:);
        if n==1; v0 = v; end
        if strcmp(plot_type,'box')
            h = boxplot(v, 'width', 0.5, 'position', n, 'color', cc{n});
            yl = setBoxStyle(h,1);
            ma = yl(2)+2; yma = max(yma, yl(2)); ymi = min(ymi, yl(1));
        elseif strcmp(plot_type,'scatter')
            errorbar(n, nanmean(v), nanstd(v)/sqrt(size(v,1)), 'Marker', 'o',...
                'MarkerSize', mksz, 'color', cc{n}, 'CapSize',4, 'linewidth', lw,...
                'MarkerFaceColor', cc{n}, 'MarkerEdgeColor', cc{n});
            ma = nanmean(v)+nanstd(v)/sqrt(length(v))+0.25; 
            yma = max(yma, ma); ymi = min(ymi, nanmean(v)-nanstd(v)/sqrt(length(v))-0.25);
        elseif strcmp(plot_type,'violin')
            violin(v, 'x', n, 'facecolor', cc{n}, 'facealpha', r,...
                'mc', 'k', 'medc', [], 'plotlegend', 0, 'edgecolor', []);
            ma = quantile(v(:), 0.99)+0.2; yma = max(yma, ma); ymi = min(ymi, min(v(:))-0.1);
        end
    end
    try; pval = ranksum(v0, v);
    catch ME; pval = NaN;
    end
    plot_pval_star(n,ma,pval);
    xlim([0.5 3.5]); box off
    set(gca,'xtick',1:3,'xticklabel',lstr,'xticklabelrotation',45);
    ylabel('Lick rate (Hz)'); ylim([ymi yma+2]);
    title(trial_str_combined{trial(t)}, 'FontWeight','Normal');
end
% set_figure_style(gcf);

%% quantify response time
idx_set = exp_idx<=2 | ((exp_idx==3|exp_idx==4) & beh_rate>70);
idx_set = idx_set & sess_len>=80;
trial = [1,2]; 
port_idx = [1,2]; % 1, correct port; 2, incorrect port

plot_type = 'violin';
lstr = {'Naive', 'Learning', 'Expert'};
cc = {mycc.gray, (mycc.gray+mycc.orange)/2, mycc.orange};

tex_onset = standard_ts_fr{3}(1) - standard_ts_fr{2}(1);
report_onset = standard_ts_fr{4}(1) - standard_ts_fr{2}(1);
figure; set(gcf,'color','w'); hold on; r = 0.7;
set(gcf, 'position', [1014 354 180 337]);
for t = 1:length(trial)
    subplot(length(trial),1,t); hold on; vdata = cell(1,3);
    yma = -Inf; ymi = Inf;
    for n = 1:3
        if n==1; idx = idx_set & beh_rate<55;
        elseif n==2; idx = idx_set & beh_rate>55 & beh_rate<75;
        elseif n==3; idx = idx_set & beh_rate>75;
        end
         
        v = lick_rate(idx,trial(t),port_idx(t));
        v = cellfun(@(x) out2(@max, x>0.1,[],2)/standard_fr, v, 'uniformoutput', 0);
        v = cellfun(@(x) nanmean(x), v);
        v = v(:);
        vdata{n} = v;
        if n==1; v0 = v; end
        if strcmp(plot_type,'box')
            h = boxplot(v, 'width', 0.5, 'position', n, 'color', cc{n});
            yl = setBoxStyle(h,1);
            ma = yl(2)+2; yma = max(yma, yl(2)); ymi = min(ymi, yl(1));
        elseif strcmp(plot_type,'scatter')
            errorbar(n, nanmean(v), nanstd(v)/sqrt(size(v,1)), 'Marker', 'o',...
                'MarkerSize', mksz, 'color', cc{n}, 'CapSize',4, 'linewidth', lw,...
                'MarkerFaceColor', cc{n}, 'MarkerEdgeColor', cc{n});
            ma = nanmean(v)+nanstd(v)/sqrt(length(v))+0.25; 
            yma = max(yma, ma); ymi = min(ymi, nanmean(v)-nanstd(v)/sqrt(length(v))-0.25);
        elseif strcmp(plot_type,'violin')
            violin(v, 'x', n, 'facecolor', cc{n}, 'facealpha', r,...
                'mc', 'k', 'medc', [], 'plotlegend', 0, 'edgecolor', []);
            ma = quantile(v(:), 0.99)+0.2; yma = max(yma, ma); ymi = min(ymi, min(v(:))-0.1);
        end
    end
    try; pval = ranksum(v0, v);
    catch ME; pval = NaN;
    end
    plot_pval_star(n,ma,pval);
    plot([0 4], tex_onset*[1 1], 'k:');
    plot([0 4], report_onset*[1 1], 'k:');
    xlim([0.5 3.5]); ylim([0 4]); box off
    set(gca,'xtick',1:3,'xticklabel',lstr,'xticklabelrotation',45);
    ylabel('Response time (s)');
    title(trial_str_combined{trial(t)}, 'FontWeight','Normal');
end
% set_figure_style(gcf);

