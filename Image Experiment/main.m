    %% main.m — ISTA / FISTA 图像去模糊对比主程序


clc; clear; close all;

%% 1. 参数与算法配置

% 模糊图像数据文件
input_file  = 'blurred.mat';

% 去模糊参数
lam         = 2e-5;             % L1 正则化系数
max_iter    = 200;              % 最大迭代次数
tol         = 1e-8;             % 收敛阈值（相对变化量）
levels      = 3;                % 小波分解级数
snap_iters  = [];  % 快照迭代步
target_f    = [];
verbose     = true;             % 是否打印迭代日志
csv_max_iter = [];             % CSV 导出的最大迭代次数（空 = 全部）

% 算法配置数组：struct('name', 显示名, 'func', @句柄, 'params', {参数})

alg_configs = {
    struct('name', 'ISTA',    'func', @ista,    'params', {{lam, max_iter, tol, snap_iters, levels, target_f, verbose}})
    struct('name', 'FISTA',   'func', @fista,   'params', {{lam, max_iter, tol, snap_iters, levels, target_f, verbose}})
};

num_algs = length(alg_configs);

%% 2. 加载模糊图像数据

% 确保 shared 类在路径中
if ~exist('shared', 'class')
    addpath(fileparts(mfilename('fullpath')));
end

if ~exist(input_file, 'file')
    error('文件 "%s" 不存在。请先运行 blur_image.m 生成模糊图像。', input_file);
end

load(input_file, 'clean', 'blurred', 'kernel', 'levels');
fprintf('  已加载 %s\n', input_file);

psnr_blur = shared.compute_psnr(clean, blurred);
ssim_blur = shared.compute_ssim(clean, blurred);
fprintf('  图像尺寸: %dx%d\n', size(clean,1), size(clean,2));
fprintf('  模糊图质量: PSNR=%.2f dB  SSIM=%.4f\n', psnr_blur, ssim_blur);

%% 3. 依次运行所有算法

fprintf('\n═══════════════════════════════════════════════════\n');
imgs = cell(1, num_algs);
results = cell(1, num_algs);
time_vals = zeros(1, num_algs);

for i = 1:num_algs
    cfg = alg_configs{i};
    fprintf('── %s ──────────────────────────────────────────\n', cfg.name);
    tic;
    [imgs{i}, results{i}] = cfg.func(blurred, kernel, cfg.params{:});
    time_vals(i) = toc;
end

%% 4. 计算评价指标

psnr_vals = zeros(1, num_algs);
ssim_vals = zeros(1, num_algs);
iter_vals = zeros(1, num_algs);

for i = 1:num_algs
    psnr_vals(i) = shared.compute_psnr(clean, imgs{i});
    ssim_vals(i) = shared.compute_ssim(clean, imgs{i});
    iter_vals(i) = length(results{i}.losses);
end

% 打印汇总表
fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('  %-12s  %8s  %8s  %10s  %8s\n', '算法', '迭代数', 'PSNR(dB)', 'SSIM', '耗时(s)');
fprintf('───────────────────────────────────────────────────────────\n');
for i = 1:num_algs
    fprintf('  %-12s  %8d  %8.2f  %10.4f  %8.3f\n', ...
        alg_configs{i}.name, iter_vals(i), psnr_vals(i), ssim_vals(i), time_vals(i));
end
fprintf('───────────────────────────────────────────────────────────\n');
fprintf('  %-12s  %8s  %8.2f  %10.4f  %8s\n', ...
    '模糊图', '-', psnr_blur, ssim_blur, '-');
fprintf('═══════════════════════════════════════════════════════════\n');

%% 5. 计算目标函数近似最优值 F*

all_final = cellfun(@(r) r.losses(end), results);
f_star = min(all_final);

%% 6. 保存结果

output_dir = 'Results001';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% 导出收敛误差 CSV
max_iters_all = max(cellfun(@(r) length(r.losses), results));
if ~isempty(csv_max_iter)
    csv_rows = min(max_iters_all, csv_max_iter);
else
    csv_rows = max_iters_all;
end
csv_path = fullfile(output_dir, 'convergence_data.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'Iteration');
for i = 1:num_algs, fprintf(fid, ',%s', alg_configs{i}.name); end
fprintf(fid, '\n');
for k = 1:csv_rows
    fprintf(fid, '%d', k);
    for i = 1:num_algs
        if k <= length(results{i}.losses)
            fprintf(fid, ',%.15e', results{i}.losses(k) - f_star);
        else
            fprintf(fid, ',NaN');
        end
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('  收敛数据已保存至 %s\n', csv_path);

% 导出收敛误差+耗时 CSV
csv_path2 = fullfile(output_dir, 'convergence_detail.csv');
fid = fopen(csv_path2, 'w');
fprintf(fid, 'Iteration');
for i = 1:num_algs
    name = alg_configs{i}.name;
    fprintf(fid, ',%s_FobjErr,%s_Time', name, name);
end
fprintf(fid, '\n');
for k = 1:csv_rows
    fprintf(fid, '%d', k);
    for i = 1:num_algs
        if k <= length(results{i}.losses)
            fprintf(fid, ',%.15e,%.6e', results{i}.losses(k) - f_star, results{i}.times(k));
        else
            fprintf(fid, ',NaN,NaN');
        end
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('  详细收敛数据已保存至 %s\n', csv_path2);

% 保存原始/模糊/重建图像
imwrite(uint8(clean * 255),   fullfile(output_dir, 'clean_gray.png'));
imwrite(uint8(blurred * 255), fullfile(output_dir, 'blurred.png'));
for i = 1:num_algs
    fname = sprintf('%s_restored.png', lower(strrep(alg_configs{i}.name, '-', '_')));
    imwrite(uint8(imgs{i} * 255), fullfile(output_dir, fname));
end

% 保存快照图像
snap_dir = fullfile(output_dir, 'snapshots');
if ~exist(snap_dir, 'dir'), mkdir(snap_dir); end
for i = 1:num_algs
    if ~isempty(results{i}.snaps)
        sn = fieldnames(results{i}.snaps);
        for j = 1:length(sn)
            snap_img = results{i}.snaps.(sn{j});
            iter_num = str2double(sn{j}(5:end));
            fname = sprintf('%s_iter%04d.png', lower(strrep(alg_configs{i}.name, '-', '_')), iter_num);
            imwrite(uint8(snap_img * 255), fullfile(snap_dir, fname));
        end
    end
end
fprintf('  快照图像已保存至 %s/ 目录\n', snap_dir);

% 保存 MAT 数据
save_vars = {'clean', 'blurred', 'kernel', 'lam', 'max_iter', 'levels', ...
             'psnr_vals', 'ssim_vals', 'time_vals', 'iter_vals'};

for i = 1:num_algs
    % 动态生成合法变量名
    var_img = matlab.lang.makeValidName(['img_' alg_configs{i}.name]);
    var_res = matlab.lang.makeValidName(['res_' alg_configs{i}.name]);
    eval(sprintf('%s = imgs{%d};', var_img, i));
    eval(sprintf('%s = results{%d};', var_res, i));
    save_vars{end+1} = var_img;
    save_vars{end+1} = var_res;
end

save(fullfile(output_dir, 'results.mat'), save_vars{:});

fprintf('\n  所有结果已保存至 %s/ 目录\n', output_dir);
fprintf('完成\n');
