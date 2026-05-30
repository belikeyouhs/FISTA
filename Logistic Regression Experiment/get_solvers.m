function s = get_solvers()
%GET_SOLVERS 返回包含所有求解器函数句柄的结构体。
%
% 提供以下五种近端梯度类算法：
%   fista                 - 固定步长的标准 FISTA
%   fista_bt              - 带自适应回溯线搜索的 FISTA
%   fista_restart_gradient - 带梯度重启策略的 FISTA
%   apfista               - 参数化加速 FISTA（APFISTA）
%   free_fista            - 自动重启 + 自适应回溯的 Free-FISTA
s.fista = @fista_solver;
s.fista_bt = @fista_bt_solver;
s.fista_restart_gradient = @fista_restart_gradient_solver;
s.apfista = @apfista_solver;
s.free_fista = @free_fista_solver;
end

%% ======================== 求解器 ========================

function result = fista_solver(problem, opts)
%FISTA_SOLVER 固定步长的标准 FISTA（快速迭代收缩-阈值算法）。
%
% 算法原理：
%   求解 min_x f(x) + h(x)，其中 f 光滑、h 非光滑但近端算子易计算。
%   每步迭代为：
%     x_{k+1} = prox_{s*h}(y_k - s * grad f(y_k))
%     t_{k+1} = (1 + sqrt(1 + 4*t_k^2)) / 2
%     y_{k+1} = x_{k+1} + (t_k - 1)/t_{k+1} * (x_{k+1} - x_k)
%   其中 s = 1/L 为固定步长，L 为 Lipschitz 常数上界。
%   相比ISTA的 O(1/k^2) 收敛速率，FISTA 达到 O(1/k^2) 加速收敛速率。
%
% 输入:
%   problem - 问题结构体（含 grad, prox_h, F, x0, L_upper 等）
%   opts    - 选项结构体（含 name, step, maxIter, tol, alpha）
%
% 输出:
%   result  - 结构体，含 x, cost, time, Lhist 等字段

opts = default_solver_opts(opts);
if isempty(opts.step)
    opts.step = 1 / problem.L_upper;    % 默认步长 s = 1/L
end

core = fista_core(problem.x0, opts.step, opts.maxIter, opts.tol, ...
    problem.grad, problem.prox_h, opts.alpha, problem.F, problem.exit_crit, false);

result = struct();
result.name = opts.name;                % 算法名称
result.x = core.x;                     % 最终迭代点
result.cost = core.cost;               % 每步目标函数值序列
result.time = core.time;               % 每步累计耗时序列
result.restartIters = [];              % 无重启
result.Lhist = repmat(1 / opts.step, numel(core.cost), 1);
% Lhist：Lipschitz 常数估计序列（固定步长，恒为 L）
result.btIters = zeros(max(numel(core.cost), 1), 1);
% btIters：回溯迭代次数序列（无回溯，恒为 0）
result.iter = max(numel(core.cost) - 1, 0);  % 总迭代次数
end

function result = fista_bt_solver(problem, opts)
%FISTA_BT_SOLVER 带自适应回溯线搜索的 FISTA。
%
% 算法原理：
%   在标准 FISTA 基础上引入回溯线搜索，自适应确定步长。
%   令 L_k 为第 k 步的 Lipschitz 估计，步长 s_k = 1/L_k。
%   回溯条件基于 Bregman 距离：
%     f(x) <= f(y) + <grad f(y), x-y> + (1/(2*s)) * ||x-y||^2
%   若不满足，则缩小步长：L <- L / rho（rho < 1，步长缩小）。
%   若满足且有余量，则放大步长：L <- L * delta（delta < 1，步长放大）。
%   该策略无需知道精确的 Lipschitz 常数。
%
% 输入:
%   problem - 问题结构体
%   opts    - 选项结构体（含 L0, rho, delta, maxIter, tol）

opts = default_solver_opts(opts);
if isempty(opts.L0)
    opts.L0 = problem.L_upper;          % 默认初始 Lipschitz 估计
end

core = fista_bt_core(problem.x0, opts.L0, opts.rho, opts.delta, opts.maxIter, ...
    opts.tol, problem.f, problem.grad, problem.prox_h, problem.h, ...
    problem.exit_crit, false, true);

result = struct();
result.name = opts.name;                % 算法名称
result.x = core.x;                     % 最终迭代点
result.cost = core.cost;               % 每步目标函数值序列
result.time = core.time;               % 每步累计耗时序列
result.restartIters = [];              % 无重启
result.Lhist = core.Lhist;             % Lipschitz 常数估计序列
result.btIters = core.btIters;         % 每步回溯尝试次数
result.iter = max(numel(core.cost) - 1, 0);
end

function result = fista_restart_gradient_solver(problem, opts)
%FISTA_RESTART_GRADIENT_SOLVER 带经典梯度重启策略的 FISTA。
%
% 算法原理：
%   标准 FISTA 的动量可能导致振荡。梯度重启策略通过检测
%   动量方向与梯度方向的对齐性来决定是否重启：
%     若 <y_k - x_k, x_k - x_{k-1}> > 0，
%   说明动量方向与前一步位移方向夹角为锐角，可能引起振荡，
%   此时重置动量参数 t = 1, y = x（即退化为一步 ISTA）。
%   该策略在目标函数局部曲率变化较大时尤为有效。
%
% 输入:
%   problem - 问题结构体
%   opts    - 选项结构体（含 step, maxIter, tol）

opts = default_solver_opts(opts);
if isempty(opts.step)
    opts.step = 1 / problem.L_upper;
end

s = opts.step;                          % 步长 = 1/L
F = problem.F;                          % 目标函数
grad = problem.grad;                    % 光滑部分梯度
prox_h = problem.prox_h;               % 近端算子
exit_crit = problem.exit_crit;         % 停止准则

x = problem.x0;                         % 当前迭代点
y = x;                                  % 外推点（含动量）
t = 1;                                  % 动量参数

cost = F(x);                            % 目标函数值记录
time = 0;                               % 累计耗时记录
restartIters = [];                      % 重启发生的迭代编号
tStart = tic;
outFlag = false;

for k = 1:opts.maxIter
    x_prev = x;                         % 保存前一步迭代点
    x = forward_backward_step(y, s, prox_h, grad);
    % x = prox_{s*h}(y - s * grad f(y))

    t_next = (1 + sqrt(1 + 4 * t^2)) / 2;
    % t_{k+1}：标准 FISTA 动量参数更新

    beta = (t - 1) / t_next;
    % beta_k：动量系数

    y_candidate = x + beta * (x - x_prev);
    % y_{k+1}：候选外推点

    if dot(y - x, x - x_prev) > 0
        % 梯度重启条件：<y_k - x_k, x_k - x_{k-1}> > 0
        % 动量方向与位移方向夹角为锐角，重启
        t = 1;
        y = x;
        restartIters(end+1, 1) = k; %#ok<AGROW>
    else
        t = t_next;
        y = y_candidate;
    end

    time(end+1, 1) = toc(tStart); %#ok<AGROW>
    cost(end+1, 1) = F(x); %#ok<AGROW>

    % 停止准则基于相邻原始迭代点 ||x_k - x_{k-1}||。
    % 注意：重启时 y 被重置为 x，若用 exit_crit(x, y) 会导致
    % 重启后立即满足停止条件，因此必须用 x 与 x_prev 比较。
    outFlag = exit_crit(x, x_prev) < opts.tol;
    if outFlag
        break;
    end
end

result = struct();
result.name = opts.name;                % 算法名称
result.x = x;                           % 最终迭代点
result.cost = cost;                     % 目标函数值序列
result.time = time;                     % 耗时序列
result.restartIters = restartIters;     % 重启迭代编号
result.Lhist = repmat(1 / s, numel(cost), 1);
result.btIters = zeros(max(numel(cost), 1), 1);
result.iter = max(numel(cost) - 1, 0);
end

function result = apfista_solver(problem, opts)
%APFISTA_SOLVER 参数化加速 FISTA（APFISTA）。
%
% 算法原理：
%   使用参数化的动量更新规则替代标准 FISTA 的固定公式：
%     t_{k+1} = (p + sqrt(q + r * t_k^2)) / d
%     beta_k  = (t_k - 1) / t_{k+1}
%     y_{k+1} = x_{k+1} + beta_k * (x_{k+1} - x_k)
%   通过调节参数 (p, q, r, d) 可控制动量衰减速率。
%   取 p=1, q=1, r=4, d=2, t0=1 时退化为标准 FISTA。
%   增大 r 或减小 d 可加速动量增长，从而加速收敛；
%   但过大的动量可能导致振荡。
%
% 输入:
%   problem - 问题结构体
%   opts    - 选项结构体（含 step, p, q, r, d, t0, maxIter, tol）

opts = default_solver_opts(opts);
if isempty(opts.step)
    opts.step = 1 / problem.L_upper;
end
% 参数默认值（标准 FISTA 对应 p=1, q=1, r=4, d=2, t0=1）
if ~isfield(opts, 'p') || isempty(opts.p), opts.p = 1; end
if ~isfield(opts, 'q') || isempty(opts.q), opts.q = 1; end
if ~isfield(opts, 'r') || isempty(opts.r), opts.r = 4; end
if ~isfield(opts, 'd') || isempty(opts.d), opts.d = 2; end
if ~isfield(opts, 't0') || isempty(opts.t0), opts.t0 = 1; end

s = opts.step;                          % 步长 = 1/L
F = problem.F;                          % 目标函数
grad = problem.grad;                    % 光滑部分梯度
prox_h = problem.prox_h;               % 近端算子
exit_crit = problem.exit_crit;         % 停止准则

x = problem.x0;                         % 当前迭代点
y = x;                                  % 外推点
t = opts.t0;                            % 动量参数初始值

cost = F(x);                            % 目标函数值记录
time = 0;                               % 累计耗时记录
outFlag = false;
tStart = tic;

for k = 1:opts.maxIter
    x_prev = x;
    x = forward_backward_step(y, s, prox_h, grad);
    % x = prox_{s*h}(y - s * grad f(y))

    t_next = (opts.p + sqrt(max(opts.q + opts.r * t^2, eps))) / opts.d;
    % t_{k+1}：参数化动量参数更新

    beta = (t - 1) / max(t_next, eps);
    % beta_k：动量系数

    y = x + beta * (x - x_prev);
    % y_{k+1}：外推点

    t = t_next;

    time(end+1, 1) = toc(tStart); %#ok<AGROW>
    cost(end+1, 1) = F(x); %#ok<AGROW>

    % 停止准则：使用相邻原始迭代点 ||x_k - x_{k-1}||。
    % 对于加速方法，若使用 exit_crit(x, y)，当 y = x（首步或
    % 重启后）时会立即触发停止，导致过早终止。
    outFlag = exit_crit(x, x_prev) < opts.tol;
    if outFlag
        break;
    end
end

result = struct();
result.name = opts.name;                % 算法名称
result.x = x;                           % 最终迭代点
result.cost = cost;                     % 目标函数值序列
result.time = time;                     % 耗时序列
result.restartIters = [];              % 无重启机制
result.Lhist = repmat(1 / s, numel(cost), 1);
result.btIters = zeros(max(numel(cost), 1), 1);
result.iter = max(numel(cost) - 1, 0);
result.param = struct('p', opts.p, 'q', opts.q, 'r', opts.r, 'd', opts.d, 't0', opts.t0);
% param：记录使用的动量参数，便于结果复现
end

function result = free_fista_solver(problem, opts)
%FREE_FISTA_SOLVER 自动重启 + 自适应回溯的 Free-FISTA。
%
% 算法原理：
%   结合自适应回溯线搜索与自动重启策略，无需手动调节步长和重启频率。
%   核心思想：
%   1. 将整个求解过程分为若干段（segment），每段运行 fista_bt_core；
%   2. 每段结束后，利用目标函数值序列估计收敛速率 q；
%   3. 若当前段长度 n 不超过 C/sqrt(q)，则将段长度加倍；
%      C = 6.38 / sqrt(rho) 为理论常数。
%   4. 当目标函数值出现上升时自动重启（重置动量）。
%   该策略在无需调节步长的条件下达到最优收敛速率。
%
% 输入:
%   problem - 问题结构体
%   opts    - 选项结构体（含 L0, rho, delta, C, maxIter, tol）

opts = default_solver_opts(opts);
if isempty(opts.L0)
    opts.L0 = problem.L_upper;          % 初始 Lipschitz 估计
end
if isempty(opts.C)
    opts.C = 6.38 / sqrt(opts.rho);    % 段长度调节常数
end
estimated_ratio = 1;                    % 收敛速率估计的初始比率

x = problem.x0;                         % 当前迭代点
maxIter = opts.maxIter;                 % 最大迭代次数

n = max(1, floor(2 * opts.C * sqrt(estimated_ratio)));
% n：当前段长度（初始值）

obj_estimate = problem.F(x);           % 各段末目标函数值记录
n_tab = n;                              % 各段长度记录
i = 0;                                  % 全局迭代计数
restartIters = 0;                       % 重启点迭代编号记录
conditionHist = [];                     % 收敛速率 q 的历史记录
cost = problem.F(x);                   % 目标函数值序列
time = 0;                               % 累计耗时序列
Lhist = opts.L0;                       % Lipschitz 常数估计序列
btIters = 0;                            % 回溯迭代次数序列

% 第一段：运行带自适应回溯的 FISTA 子例程
seg = fista_bt_core(x, opts.L0, opts.rho, opts.delta, min(n, maxIter - i), ...
    opts.tol, problem.f, problem.grad, problem.prox_h, problem.h, ...
    problem.exit_crit, true, true);

x = seg.x;                              % 更新迭代点
outFlag = seg.out;                      % 是否已满足停止条件
L_tilde = seg.Lhist(end);              % 段末 Lipschitz 常数估计
i = i + min(n, maxIter - i);           % 更新全局迭代计数
cost = [cost; seg.cost(:)];            % 追加目标函数值
time = [time; seg.time(2:end) + time(end)];
% 追加耗时（偏移使时间连续）
Lhist = [Lhist; seg.Lhist(:)];
btIters = [btIters; seg.btIters(:)];
obj_estimate(end+1, 1) = seg.fx + problem.h(x);
% 记录段末完整目标函数值 F(x) = f(x) + h(x)
n_tab(end+1, 1) = n;

% 主循环：反复执行段式 FISTA + 自适应段长度调节
while (i < maxIter) && ~outFlag
    restartIters(end+1, 1) = i; %#ok<AGROW>
    % 记录重启点

    segLen = min(n, maxIter - i);       % 当前段实际长度
    seg = fista_bt_core(x, L_tilde, opts.rho, opts.delta, segLen, opts.tol, ...
        problem.f, problem.grad, problem.prox_h, problem.h, ...
        problem.exit_crit, true, true);

    x = seg.x;
    outFlag = seg.out;
    L_tilde = seg.Lhist(end);
    i = i + segLen;

    cost = [cost; seg.cost(:)]; %#ok<AGROW>
    time = [time; seg.time(2:end) + time(end)]; %#ok<AGROW>
    Lhist = [Lhist; seg.Lhist(:)]; %#ok<AGROW>
    btIters = [btIters; seg.btIters(:)]; %#ok<AGROW>
    obj_estimate(end+1, 1) = seg.fx + problem.h(x); %#ok<AGROW>

    % 自适应段长度调节
    if numel(obj_estimate) >= 3
        % 估计收敛速率 q
        % q_k = min over j of [ 4/(rho*(n_j+1)^2) * (F_j - F_end) / (F_{j+1} - F_end) ]
        tab_q = (4 ./ (opts.rho * (n_tab(1:end-1) + 1).^2)) .* ...
            ((obj_estimate(1:end-2) - obj_estimate(end)) ./ ...
             max(obj_estimate(2:end-1) - obj_estimate(end), eps));
        q = min(tab_q);
        conditionHist(end+1, 1) = q; %#ok<AGROW>
        if n <= opts.C / sqrt(max(q, eps))
            n = 2 * n;                  % 段长度加倍
        end
    end
    n_tab(end+1, 1) = n; %#ok<AGROW>
end

result = struct();
result.name = opts.name;                % 算法名称
result.x = x;                           % 最终迭代点
result.cost = cost;                     % 目标函数值序列
result.time = time;                     % 耗时序列
result.restartIters = restartIters;     % 重启点迭代编号
result.Lhist = Lhist;                   % Lipschitz 常数估计序列
result.btIters = btIters;              % 回溯迭代次数序列
result.conditionHist = conditionHist;  % 收敛速率 q 的历史
result.iter = max(numel(cost) - 1, 0);
end

%% ======================== 核心迭代引擎 ========================

function out = fista_core(x, s, maxIter, tol, grad, prox_h, alpha, F, exit_crit, restarted)
%FISTA_CORE FISTA 核心迭代。
%
% 算法原理：
%   复合梯度步 + Nesterov 动量加速：
%     x_{k+1} = prox_{s*h}(y_k - s * grad f(y_k))
%     y_{k+1} = x_{k+1} + (k-1)/(k+alpha-1) * (x_{k+1} - x_k)
%   其中 alpha >= 3 控制动量衰减速率（alpha=3 对应标准 FISTA）。
%
% 输入:
%   x         - 初始点（n x 1）
%   s         - 步长（标量）
%   maxIter   - 最大迭代次数
%   tol       - 停止阈值
%   grad      - 光滑部分梯度句柄
%   prox_h    - 近端算子句柄
%   alpha     - 动量衰减参数（默认 3）
%   F         - 完整目标函数句柄（用于记录函数值）
%   exit_crit - 停止准则句柄
%   restarted - 是否作为子例程被调用（影响函数值记录方式）
%
% 输出:
%   out - 结构体，含 x, cost, time, out

if nargin < 10 || isempty(restarted), restarted = false; end
if nargin < 9 || isempty(exit_crit), exit_crit = @(xk, yk) norm(xk - yk, 2); end
if nargin < 8 || isempty(F), F = []; end
if nargin < 7 || isempty(alpha), alpha = 3; end

if ~isempty(F)
    if restarted
        cost = [];                      % 作为子例程时不记录首步函数值
    else
        cost = F(x);
    end
else
    cost = [];
end
ctime = 0;                              % 累计耗时

y = x;                                  % 外推点
outFlag = false;
tStart = tic;

for k = 1:maxIter
    x_prev = x;
    x = forward_backward_step(y, s, prox_h, grad);
    % x_{k+1} = prox_{s*h}(y_k - s * grad f(y_k))

    outFlag = exit_crit(x, y) < tol;

    y = x + (k - 1) / (k + alpha - 1) * (x - x_prev);
    % y_{k+1} = x_{k+1} + ((k-1)/(k+alpha-1)) * (x_{k+1} - x_k)
    % 动量系数随迭代递减，保证收敛

    ctime(end+1, 1) = toc(tStart); %#ok<AGROW>
    if ~isempty(F)
        cost(end+1, 1) = F(x); %#ok<AGROW>
    end
    if outFlag
        break;
    end
end

out = struct('x', x, 'cost', cost, 'time', ctime, 'out', outFlag);
end

function out = fista_bt_core(x, L0, rho, delta, maxIter, tol, f, grad, prox_h, h, exit_crit, restarted, exit_norm)
%FISTA_BT_CORE 带自适应回溯线搜索的 FISTA 核心迭代。
%
% 算法原理：
%   在每步迭代中，通过回溯线搜索自适应确定步长 s_k = 1/L_k：
%   1. 前向步：放大步长（s <- s/delta, delta<1 即 L 缩小）；
%   2. 回溯循环：若 Bregman 距离条件不满足，则缩小步长：
%        s <- rho * s （即 L <- L/rho, rho<1 即 L 增大）
%      直到满足：
%        f(x) <= f(y) + <grad f(y), x-y> + (1/(2*s)) * ||x-y||^2
%   3. 若条件满足且有余量（Bregman 距离 <= rho * 二次上界），
%      则标记为"前向"步，下一步继续放大步长。
%
% 输入:
%   x         - 初始点（n x 1）
%   L0        - 初始 Lipschitz 常数估计
%   rho       - 回溯缩小因子（0 < rho < 1）
%   delta     - 前向放大因子（0 < delta < 1）
%   maxIter   - 最大迭代次数
%   tol       - 停止阈值
%   f         - 光滑部分函数值句柄
%   grad      - 光滑部分梯度句柄
%   prox_h    - 近端算子句柄
%   h         - 非光滑部分函数值句柄
%   exit_crit - 停止准则句柄
%   restarted - 是否作为子例程被调用
%   exit_norm - 是否使用 ||x-y|| 作为停止准则
%
% 输出:
%   out - 结构体，含 x, cost, time, Lhist, btIters, fx, out

if nargin < 13 || isempty(exit_norm), exit_norm = true; end
if nargin < 12 || isempty(restarted), restarted = false; end
if nargin < 11 || isempty(exit_crit)
    exit_crit = @(xk, yk) norm(xk - yk, 2);
    exit_norm = true;
end
if nargin < 10, h = []; end

if ~isempty(h) && ~restarted
    cost = f(x) + h(x);                % 首步完整目标函数值
elseif ~isempty(h) && restarted
    cost = [];
else
    cost = [];
end

if restarted
    Lhist = [];                         % 作为子例程时由外层管理
    btIters = [];
else
    Lhist = L0;                         % 初始 Lipschitz 估计
    btIters = 0;
end
ctime = 0;                              % 累计耗时

s = 1 / L0;                            % 初始步长
t = 1;                                  % 动量参数
k = 0;                                  % 迭代计数
x_prev = x;                            % 前一步迭代点
outFlag = false;
forward = true;                         % 是否为前向步（步长放大标志）
tStart = tic;
last_fx = f(x);                        % 光滑部分函数值

while k < maxIter && ~outFlag
    if forward
        s = s / delta;                  % 前向步：放大步长（缩小 L 估计）
    end

    % 回溯线搜索循环
    cond = false;
    j = 0;                              % 回溯尝试次数
    while ~cond
        temp_s = (rho ^ j) * s;        % 候选步长 s * rho^j
        temp_t = (1 + sqrt(1 + 4 * t^2 * s / temp_s)) / 2;
        % 动量参数，考虑步长变化：t_{k+1} = (1 + sqrt(1 + 4*t_k^2*s/s_new)) / 2

        temp_beta = (t - 1) / temp_t;  % 动量系数
        temp_y = x + temp_beta * (x - x_prev);
        % 外推点

        temp_x = forward_backward_step(temp_y, temp_s, prox_h, grad);
        % 近端梯度步

        diff = temp_x - temp_y;
        temp_norm = diff' * diff;       % ||x - y||^2

        [temp_breg, fx] = bregman_distance(temp_x, temp_y, f, grad);
        % Bregman 距离：D_f(x,y) = f(x) - f(y) - <grad f(y), x-y>

        cond = temp_breg <= temp_norm / (2 * temp_s);
        % 回溯条件：D_f(x,y) <= (1/(2*s)) * ||x-y||^2
        j = j + 1;
    end

    btIters(end+1, 1) = j; %#ok<AGROW>

    forward = temp_breg <= rho * temp_norm / (2 * temp_s);
    % 前向步判定：若 Bregman 距离有足够余量（<= rho * 二次上界），
    % 则下一步继续放大步长

    s = temp_s;                         % 更新步长
    t = temp_t;                         % 更新动量参数
    x_prev = x;
    x = temp_x;
    last_fx = fx;

    if exit_norm
        outFlag = sqrt(temp_norm) < tol;
    else
        outFlag = exit_crit(x, temp_y) < tol;
    end

    ctime(end+1, 1) = toc(tStart); %#ok<AGROW>
    if ~isempty(h)
        cost(end+1, 1) = fx + h(x); %#ok<AGROW>
    end
    Lhist(end+1, 1) = 1 / s; %#ok<AGROW>

    k = k + 1;
end

out = struct();
out.x = x;                              % 最终迭代点
out.cost = cost;                        % 目标函数值序列
out.time = ctime;                       % 耗时序列
out.Lhist = Lhist;                      % Lipschitz 常数估计序列
out.btIters = btIters;                  % 每步回溯尝试次数
out.fx = last_fx;                       % 最终光滑部分函数值
out.out = outFlag;                      % 是否满足停止条件
end

%% ======================== 工具函数 ========================

function xnew = forward_backward_step(y, s, prox_h, grad)
%FORWARD_BACKWARD_STEP 近端梯度步（前向-后向分裂）。
%   x = prox_{s*h}(y - s * grad f(y))
% 输入:
%   y      - 外推点（n x 1）
%   s      - 步长
%   prox_h - 非光滑部分的近端算子
%   grad   - 光滑部分的梯度
% 输出:
%   xnew   - 更新后的迭代点
xnew = prox_h(y - s * grad(y), s);
end

function [D, fx] = bregman_distance(x, y, f, grad)
%BREGMAN_DISTANCE 计算光滑函数 f 关于 y 的 Bregman 距离。
%   D_f(x, y) = f(x) - f(y) - <grad f(y), x - y>
%   Bregman 距离度量了 f 在 x 处相对于其在 y 处的一阶近似的偏差。
%   当 f 为二次函数时，Bregman 距离退化为加权欧氏距离。
% 输入:
%   x    - 候选点
%   y    - 参考点
%   f    - 光滑函数句柄
%   grad - 光滑函数梯度句柄
% 输出:
%   D    - Bregman 距离 D_f(x, y)
%   fx   - f(x) 的函数值
fy = f(y);
fx = f(x);
D = fx - fy - grad(y)' * (x - y);
end

function opts = default_solver_opts(opts)
%DEFAULT_SOLVER_OPTS 填充求解器选项的默认值。
% 输入/输出:
%   opts - 选项结构体，缺失字段自动填充默认值
if nargin == 0 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'name'),    opts.name = 'unnamed solver'; end   % 算法名称
if ~isfield(opts, 'maxIter'), opts.maxIter = 1000; end            % 最大迭代次数
if ~isfield(opts, 'tol'),     opts.tol = 1e-8; end               % 停止阈值
if ~isfield(opts, 'alpha'),   opts.alpha = 3; end                % 动量衰减参数
if ~isfield(opts, 'step'),    opts.step = []; end                % 固定步长（空则自动计算）
if ~isfield(opts, 'L0'),      opts.L0 = []; end                  % 初始 Lipschitz 估计
if ~isfield(opts, 'rho'),     opts.rho = 0.8; end                % 回溯缩小因子
if ~isfield(opts, 'delta'),   opts.delta = 0.95; end             % 前向放大因子
if ~isfield(opts, 'C'),       opts.C = []; end                   % 段长度调节常数
end
