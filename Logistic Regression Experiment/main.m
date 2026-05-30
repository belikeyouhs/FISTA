function main()
%MAIN L2-L1 正则化逻辑回归基准测试主程序。
%
% 对以下五种近端梯度类算法进行数值比较：
%   1. 标准 FISTA（固定步长）
%   2. 梯度重启 FISTA
%   3. 带自适应回溯线搜索的 FISTA
%   4. 参数化加速 FISTA（APFISTA）
%   5. Free-FISTA（自动重启 + 自适应回溯）
%
% 输出：
%   - 目标函数值与参考最优值之差随迭代次数/时间的变化图
%   - 各算法的 CSV 数据文件（保存至 data/ 目录）

cfg = default_reglog_config();          % 加载默认配置
problem = build_problem(cfg);           % 构建测试问题
solvers = register_default_solvers(problem, cfg);
% 注册所有求解器

results = run_solver_suite(problem, solvers);
% 运行全部求解器

% 使用 Free-FISTA 的结果作为初始解，进一步优化得到更精确的参考最优值。
% Free-FISTA 具有自适应重启和回溯机制，通常能找到较优的解。
freeIdx = numel(results);
freeRes = results{freeIdx};
[xmin, ~] = refine_reference_solution(problem, freeRes.x, freeRes.Lhist(end), ...
    cfg.bt.rho, cfg.bt.delta, cfg);
Fmin = problem.F(xmin);
% Fmin：参考最优目标函数值

% 绘制收敛曲线（目标函数值差 vs 迭代次数 / 计算时间）
plot_objective_gap(results, Fmin, true, cfg.plot.fontSize);
plot_objective_gap(results, Fmin, false, cfg.plot.fontSize);

% 保存各算法结果为 CSV 文件，便于使用 Origin 等工具绘图
dataDir = fullfile(fileparts(mfilename('fullpath')), 'data');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
for k = 1:numel(results)
    r = results{k};
    iter = (0:numel(r.cost)-1)';       % 迭代编号（从 0 开始）
    gap = r.cost - Fmin;               % 目标函数值与参考最优值之差
    T = table(iter, r.time(:), r.cost(:), gap(:), ...
        'VariableNames', {'Iteration', 'Time_sec', 'Cost', 'Gap'});
    % Cost = 当前解的函数值 F(x_k)
    % Gap  = F(x_k) - F_ref
    fname = fullfile(dataDir, [r.name, '.csv']);
    writetable(T, fname);
    fprintf('已保存: %s\n', fname);
end
end

%% ======================== 配置 ========================

function cfg = default_reglog_config()
%DEFAULT_REGLOG_CONFIG 逻辑回归基准测试的默认配置。
%
% 输出:
%   cfg - 配置结构体

cfg.seed = 1896;                        % 随机种子

% 数据维度
cfg.n = 30000;                          % 特征维度
cfg.m = 100;                            % 样本数

% 正则化参数
cfg.lambda1 = 1.0;                     % 逻辑损失缩放系数

% 迭代 / 停止条件
cfg.maxIter = 2000;                     % 最大迭代次数
cfg.tol = 1e-8;                        % 停止阈值

% 回溯线搜索参数
cfg.bt.L0_small = 1.0;                % 初始 Lipschitz 估计值
cfg.bt.rho = 0.8;                      % 回溯缩小因子（步长缩小比例）
cfg.bt.delta = 0.95;                   % 前向放大因子（步长放大比例）

% APFISTA 动量参数。
% 动量更新规则：
%   t_{k+1} = (p + sqrt(q + r * t_k^2)) / d
%   beta_k  = (t_k - 1) / t_{k+1}
% 标准 FISTA 对应 p=1, q=1, r=4, d=2, t0=1。
cfg.apfista.p = 2;                     % 分子常数项
cfg.apfista.q = 1;                     % 分子根号内常数项
cfg.apfista.r = 100;                   % 分子根号内 t_k^2 的系数
cfg.apfista.d = 10;                    % 分母
cfg.apfista.t0 = 1.0;                  % 动量参数初始值

% 参考解精细化参数
cfg.refine.maxIter = 3000;             % 精细化最大迭代次数
cfg.refine.tol = 1e-12;                % 精细化停止阈值（更严格）

% 绘图
cfg.plot.fontSize = 16;                % 图表字体大小
end

%% ======================== 求解器注册 ========================

function solvers = register_default_solvers(problem, cfg)
%REGISTER_DEFAULT_SOLVERS 注册默认算法套件。
% 输入:
%   problem - 问题结构体
%   cfg     - 配置结构体
% 输出:
%   solvers - 元胞数组，每个元素为含 name, solver, opts 的结构体

sf = get_solvers();
solvers = {
    struct('name', 'FISTA', 'solver', sf.fista, 'opts', struct( ...
        'name', 'FISTA', 'step', 1 / problem.L_upper, 'maxIter', cfg.maxIter, 'tol', cfg.tol));
    % 标准 FISTA：步长 s = 1/L

    struct('name', 'Gradient restart FISTA', 'solver', sf.fista_restart_gradient, 'opts', struct( ...
        'name', 'Gradient restart FISTA', 'step', 1 / problem.L_upper, 'maxIter', cfg.maxIter, 'tol', cfg.tol));
    % 梯度重启 FISTA

    struct('name', sprintf('FISTA with backtracking, rho=%.2f, delta=%.2f, L0=%g', cfg.bt.rho, cfg.bt.delta, cfg.bt.L0_small), ...
        'solver', sf.fista_bt, 'opts', struct( ...
        'name', sprintf('FISTA with backtracking, rho=%.2f, delta=%.2f, L0=%g', cfg.bt.rho, cfg.bt.delta, cfg.bt.L0_small), ...
        'L0', cfg.bt.L0_small, 'rho', cfg.bt.rho, 'delta', cfg.bt.delta, 'maxIter', cfg.maxIter, 'tol', cfg.tol));
    % 带自适应回溯的 FISTA

    struct('name', sprintf('APFISTA, p=%.2f, q=%.2f, r=%.2f,d=%.2f', cfg.apfista.p, cfg.apfista.q, cfg.apfista.r,cfg.apfista.d), ...
        'solver', sf.apfista, 'opts', struct( ...
        'name', sprintf('APFISTA, p=%.2f, q=%.2f, r=%.2f,d=%.2f', cfg.apfista.p, cfg.apfista.q, cfg.apfista.r,cfg.apfista.d), ...
        'step', 1 / problem.L_upper, 'p', cfg.apfista.p, 'q', cfg.apfista.q, ...
        'r', cfg.apfista.r,'d', cfg.apfista.d, 't0', cfg.apfista.t0, 'maxIter', cfg.maxIter, 'tol', cfg.tol));
    % 参数化加速 FISTA

    struct('name', sprintf('Free-FISTA, rho=%.2f, delta=%.2f, L0=%g', cfg.bt.rho, cfg.bt.delta, cfg.bt.L0_small), ...
        'solver', sf.free_fista, 'opts', struct( ...
        'name', sprintf('Free-FISTA, rho=%.2f, delta=%.2f, L0=%g', cfg.bt.rho, cfg.bt.delta, cfg.bt.L0_small), ...
        'L0', cfg.bt.L0_small, 'rho', cfg.bt.rho, 'delta', cfg.bt.delta, 'maxIter', cfg.maxIter, 'tol', cfg.tol))
    % Free-FISTA：自动重启 + 自适应回溯
};
end

%% ======================== 编排 ========================

function results = run_solver_suite(problem, solvers)
%RUN_SOLVER_SUITE 依次运行所有已注册的求解器。
% 输入:
%   problem - 问题结构体
%   solvers - 求解器注册元胞数组
% 输出:
%   results - 元胞数组，每个元素为求解器返回的结果结构体

results = cell(size(solvers));
for k = 1:numel(solvers)
    spec = solvers{k};
    fprintf('[%d/%d] Running %s ...\n', k, numel(solvers), spec.name);
    results{k} = spec.solver(problem, spec.opts);
end
end

function [xmin, refResult] = refine_reference_solution(problem, xStart, L0, rho, delta, cfg)
%REFINE_REFERENCE_SOLUTION 使用 Free-FISTA 精细化参考最优解。
%   以 Free-FISTA 的终止解为起点，用更严格的停止阈值和更多迭代次数
%   继续求解，得到更精确的参考最优解。
% 输入:
%   problem - 问题结构体
%   xStart  - 起始点（Free-FISTA 的终止解）
%   L0      - 初始 Lipschitz 估计（来自 Free-FISTA）
%   rho     - 回溯缩小因子
%   delta   - 前向放大因子
%   cfg     - 配置结构体
% 输出:
%   xmin      - 精细化后的近似最优解
%   refResult - 精细化过程的完整结果结构体

sf = get_solvers();
refProblem = problem;
refProblem.x0 = xStart;               % 以 Free-FISTA 的解为起点
refOpts = struct('name', 'Reference Free-FISTA', 'L0', L0, 'rho', rho, 'delta', delta, ...
    'maxIter', cfg.refine.maxIter, 'tol', cfg.refine.tol);
refResult = sf.free_fista(refProblem, refOpts);
xmin = refResult.x;
end

%% ======================== 绘图 ========================

function plot_objective_gap(results, Fmin, useIteration, fontSize)
%PLOT_OBJECTIVE_GAP 绘制 F(x_k) - F_ref 随迭代次数或计算时间的变化。
% 输入:
%   results      - 求解器结果元胞数组
%   Fmin         - 参考最优目标函数值
%   useIteration - true 时横轴为迭代次数，false 时横轴为计算时间（秒）
%   fontSize     - 图表字体大小

if nargin < 4, fontSize = 14; end
figure; hold on;
for k = 1:numel(results)
    r = results{k};
    if useIteration
        xaxis_vals = 0:numel(r.cost)-1;    % 迭代次数
    else
        xaxis_vals = r.time(:)';            % 计算时间（秒）
    end
    gap = r.cost - Fmin;                    % 目标函数值差
    plot(xaxis_vals, gap, 'LineWidth', 1.5, 'DisplayName', r.name);
end
if useIteration
    xlabel('迭代次数', 'FontSize', fontSize);
else
    xlabel('计算时间（秒）', 'FontSize', fontSize);
end
ylabel('F(x_k)-F_{ref}', 'FontSize', fontSize);
legend('Location', 'best');
grid on; box on;
end

function plot_L_history(results, Lref, fontSize)
%PLOT_L_HISTORY 绘制各算法估计的 Lipschitz 常数随迭代的变化。
% 输入:
%   results  - 求解器结果元胞数组
%   Lref     - 理论 Lipschitz 常数上界
%   fontSize - 图表字体大小

if nargin < 3, fontSize = 14; end
figure; hold on;
for k = 1:numel(results)
    r = results{k};
    if isfield(r, 'Lhist') && ~isempty(r.Lhist)
        plot(r.Lhist, 'LineWidth', 1.5, 'DisplayName', r.name);
    end
end
yline(Lref, '--', 'LineWidth', 1.5, 'DisplayName', '理论上界');
xlabel('迭代次数', 'FontSize', fontSize);
ylabel('Lipschitz 常数估计值', 'FontSize', fontSize);
legend('Location', 'best');
grid on; box on;
end

function plot_backtracking_bars(results, fontSize)
%PLOT_BACKTRACKING_BARS 绘制各算法每步的回溯尝试次数柱状图。
% 输入:
%   results  - 求解器结果元胞数组
%   fontSize - 图表字体大小

if nargin < 2, fontSize = 14; end
figure; hold on;
for k = 1:numel(results)
    r = results{k};
    if isfield(r, 'btIters') && ~isempty(r.btIters)
        bar(0:numel(r.btIters)-1, r.btIters, 'DisplayName', r.name, 'FaceAlpha', 0.35);
    end
end
xlabel('全局迭代次数', 'FontSize', fontSize);
ylabel('回溯迭代次数', 'FontSize', fontSize);
legend('Location', 'best');
grid on; box on;
end
