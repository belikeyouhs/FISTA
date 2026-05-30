%% param_test.m — 正则化参数 λ 扫描与收敛曲线对比
%  对不同 λ 值运行 FISTA，保存误差曲线与 PSNR/SSIM 数据表

clc; clear; close all;

%% 1. 参数配置

input_file = 'blurred.mat';

% λ 扫描值
lambda_values = [1e-7, 1e-6, 1e-5, 2e-5, 5e-5, 1e-4, 1e-3];

% FISTA 固定参数
algorithm  = @fista;
max_iter   = 500;
tol        = 1e-8;
levels     = 3;
target_f   = [];
verbose    = false;
psnr_step  = 20;
snap_iters = psnr_step:psnr_step:max_iter;

% 输出设置
output_dir   = 'results_sweep';
save_figures = true;
save_data    = true;

%% 2. 加载数据

if ~exist('shared', 'class')
    addpath(fileparts(mfilename('fullpath')));
end

if ~exist(input_file, 'file')
    error('文件 "%s" 不存在。请先运行 blur_image.m 生成模糊图像。', input_file);
end
load(input_file, 'clean', 'blurred', 'kernel', 'levels');
fprintf('已加载数据: %s\n', input_file);
fprintf('图像尺寸: %dx%d\n', size(clean,1), size(clean,2));

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

num_lam = length(lambda_values);
colors  = lines(num_lam);

losses_cell = cell(1, num_lam);
times_cell  = cell(1, num_lam);
final_imgs  = cell(1, num_lam);
snaps_cell  = cell(1, num_lam);
psnr_vals   = zeros(1, num_lam);
ssim_vals   = zeros(1, num_lam);
iter_vals   = zeros(1, num_lam);

%% 3. λ 扫描

fprintf('\n═══════════════════════════════════════════════════════════\n');
fprintf('开始 λ 扫描: %s\n', func2str(algorithm));
fprintf('λ = '); fprintf('%g ', lambda_values); fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════\n');

for p = 1:num_lam
    lam = lambda_values(p);
    fprintf('运行 λ = %.2e ... ', lam);
    tic;
    [img, res] = algorithm(blurred, kernel, lam, max_iter, tol, ...
                           snap_iters, levels, target_f, verbose);
    elapsed = toc;

    losses_cell{p} = res.losses;
    times_cell{p}  = res.times;
    final_imgs{p}  = img;
    snaps_cell{p}  = res.snaps;
    iter_vals(p)   = length(res.losses);
    psnr_vals(p)   = shared.compute_psnr(clean, img);
    ssim_vals(p)   = shared.compute_ssim(clean, img);

    fprintf('迭代 %d 步，耗时 %.2f s，PSNR = %.2f dB\n', ...
            iter_vals(p), elapsed, psnr_vals(p));
end

fprintf('═══════════════════════════════════════════════════════════\n');

%% 4. 绘制收敛曲线（误差 vs 迭代/耗时）

% 近似最优值 F*
all_final = cellfun(@(c) c(end), losses_cell);
f_star = min(all_final);

h_fig1 = figure('Name', '收敛曲线对比', 'Position', [50 50 1200 500], 'Color', 'w');

% 左：误差 vs 迭代
subplot(1,2,1);
for p = 1:num_lam
    iters = 1:length(losses_cell{p});
    errs  = losses_cell{p} - f_star;
    semilogy(iters(1:300), errs(1:300), 'LineWidth', 2.0, 'Color', colors(p,:), ...
             'DisplayName', sprintf('\\lambda = %.2e', lambda_values(p)));
    hold on;
end
xlabel('迭代次数 k'); ylabel('F(x_k) - F*（对数轴）');
title('目标函数误差 vs 迭代次数');
legend('Location', 'northeast'); grid on; box on;

% 右：误差 vs 耗时
subplot(1,2,2);
for p = 1:num_lam
    errs = losses_cell{p} - f_star;
    semilogy(times_cell{p}(1:300), errs(1:300), 'LineWidth', 2.0, 'Color', colors(p,:), ...
             'DisplayName', sprintf('\\lambda = %.2e', lambda_values(p)));
    hold on;
end
xlabel('累计耗时（秒）'); ylabel('F(x_k) - F*（对数轴）');
title('目标函数误差 vs 实际耗时');
legend('Location', 'northeast'); grid on; box on;

sgtitle('收敛曲线对比 — 不同 \lambda', 'FontSize', 14, 'FontWeight', 'bold');

%% 5. 绘制目标函数原始值曲线

h_fig2 = figure('Name', '目标函数值曲线', 'Position', [100 100 800 500], 'Color', 'w');

for p = 1:num_lam
    iters = 1:length(losses_cell{p});
    semilogy(iters, losses_cell{p}, 'LineWidth', 2.0, 'Color', colors(p,:), ...
         'DisplayName', sprintf('\\lambda = %.2e', lambda_values(p)));
    hold on;
end
xlabel('迭代次数 k'); ylabel('F(x_k)');
title('目标函数值 F(x_k) — 不同 \lambda');
legend('Location', 'northeast'); grid on; box on;

%% 6. 保存结果

base_name = sprintf('%s_sweep_lambda', func2str(algorithm));

if save_figures
    saveas(h_fig1, fullfile(output_dir, [base_name '_convergence.png']));
    saveas(h_fig2, fullfile(output_dir, [base_name '_objective.png']));
    fprintf('\n图形已保存至 %s/\n', output_dir);
end

%% 7. 导出 PSNR 随迭代步变化的 CSV

psnr_table = zeros(length(snap_iters), num_lam);
for p = 1:num_lam
    for j = 1:length(snap_iters)
        snap_field = ['iter' num2str(snap_iters(j))];
        if isfield(snaps_cell{p}, snap_field)
            psnr_table(j, p) = shared.compute_psnr(clean, snaps_cell{p}.(snap_field));
        else
            psnr_table(j, p) = NaN;
        end
    end
end

psnr_csv = fullfile(output_dir, [base_name '_psnr_table.csv']);
fid = fopen(psnr_csv, 'w');
fprintf(fid, 'Iter');
for p = 1:num_lam, fprintf(fid, ',lambda=%.2e', lambda_values(p)); end
fprintf(fid, '\n');
for j = 1:length(snap_iters)
    fprintf(fid, '%d', snap_iters(j));
    for p = 1:num_lam, fprintf(fid, ',%.4f', psnr_table(j, p)); end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('PSNR 数据表: %s\n', psnr_csv);

%% 8. 导出 SSIM 随迭代步变化的 CSV

ssim_table = zeros(length(snap_iters), num_lam);
for p = 1:num_lam
    for j = 1:length(snap_iters)
        snap_field = ['iter' num2str(snap_iters(j))];
        if isfield(snaps_cell{p}, snap_field)
            ssim_table(j, p) = shared.compute_ssim(clean, snaps_cell{p}.(snap_field));
        else
            ssim_table(j, p) = NaN;
        end
    end
end

ssim_csv = fullfile(output_dir, [base_name '_ssim_table.csv']);
fid = fopen(ssim_csv, 'w');
fprintf(fid, 'Iter');
for p = 1:num_lam, fprintf(fid, ',lambda=%.2e', lambda_values(p)); end
fprintf(fid, '\n');
for j = 1:length(snap_iters)
    fprintf(fid, '%d', snap_iters(j));
    for p = 1:num_lam, fprintf(fid, ',%.6f', ssim_table(j, p)); end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('SSIM 数据表: %s\n', ssim_csv);

%% 9. 保存 MAT 数据

if save_data
    save(fullfile(output_dir, [base_name '_data.mat']), ...
         'lambda_values', 'algorithm', ...
         'losses_cell', 'times_cell', 'final_imgs', 'snaps_cell', ...
         'psnr_vals', 'ssim_vals', 'iter_vals', ...
         'psnr_table', 'ssim_table', 'snap_iters', ...
         'clean', 'blurred', 'kernel', 'levels');
    fprintf('数据已保存至 %s/\n', output_dir);
end

fprintf('\n完成。\n');
