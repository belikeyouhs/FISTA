function p = build_problem(cfg)
%BUILD_PROBLEM 构建 L2-L1 正则化逻辑回归测试问题。
%
% 目标函数为复合优化问题：
%   min_x  F(x) = f(x) + h(x)
%
% 其中光滑部分 f(x) 为正则化逻辑损失：
%   f(x) = beta * sum_{j=1}^{m} log(1 + exp(-b_j * a_j^T * x))
%        + (lambda2 / 2) * ||x||^2
%
% 非光滑部分 h(x) 为 L1 正则项（稀疏性诱导）：
%   h(x) = ||x||_1
%
% beta = lambda1 / (2 * ||A^T b||_inf) 为归一化系数，
% lambda2 由光滑部分的 Lipschitz 常数按比例导出。
%
% 输入:
%   cfg - 配置结构体，包含 seed, m, n, lambda1 等参数
%
% 输出:
%   p - 结构体，包含问题数据和函数句柄

rng(cfg.seed);                      % 固定随机种子，保证实验可重复

m = cfg.m;                          % 样本数
n = cfg.n;                          % 特征维度
A = rand(m, n);                     % 随机生成数据矩阵 A（m x n）
b = 2 * randi([0, 1], m, 1) - 1;   % 标签向量 b，取值为 {-1, +1}
ba = bsxfun(@times, b, A);          % ba(j,:) = b_j * a_j，逐行乘以标签

lambda1 = cfg.lambda1;              % 逻辑损失缩放系数
Atb = A' * b;                       % A^T * b，用于计算归一化系数

beta = lambda1 / (2 * norm(Atb, inf));
% beta：逻辑损失的归一化系数，使得梯度量级与正则项匹配

L_log = lambda1 * norm(Atb, 2)^2 / (8 * norm(Atb, inf));
% L_log：逻辑损失部分 f_log 的 Lipschitz 常数上界
% 由 Hessian 矩阵的谱范数估计得到：L <= lambda1 * ||A^T b||_2^2 / (8 * ||A^T b||_inf)

lambda2 = L_log / (10 * n);
% lambda2：L2 正则化系数，取为 L_log 的 1/(10n)，使 L2 项量级较小

L_upper = L_log + lambda2;
% L_upper：光滑部分 f(x) 整体的 Lipschitz 常数上界

x0 = 2 * rand(n, 1) - 1;           % 初始点，各分量在 [-1, 1] 上均匀分布

% ---------- 输出结构体 ----------
p.A = A;                            % 数据矩阵（m x n）
p.b = b;                            % 标签向量（m x 1）
p.ba = ba;                          % 标签加权数据矩阵（m x n），ba = diag(b) * A
p.n = n;                            % 特征维度
p.m = m;                            % 样本数
p.lambda1 = lambda1;                % 逻辑损失缩放系数
p.lambda2 = lambda2;                % L2 正则化系数
p.beta = beta;                      % 逻辑损失归一化系数
p.L_upper = L_upper;                % 光滑部分 Lipschitz 常数上界
p.x0 = x0;                         % 初始点
p.exit_crit = @(x, xprev) norm(x - xprev, 2);
% 停止准则：相邻迭代点的欧氏距离

% ---------- 函数句柄 ----------
p.f = @(x) smooth_part(x, ba, beta, lambda2);
% f(x)：光滑部分（逻辑损失 + L2 正则项）

p.h = @(x) norm(x, 1);
% h(x)：非光滑部分（L1 范数）

p.F = @(x) p.f(x) + p.h(x);
% F(x)：完整目标函数

p.grad = @(x) smooth_grad(x, ba, beta, lambda2);
% grad f(x)：光滑部分的梯度

p.prox_h = @(x, s) soft_threshold(x, s);
% prox_{s*h}(x)：L1 范数的近端算子（软阈值）
end

%% ======================== 局部函数 ========================

function val = smooth_part(x, ba, beta, lambda2)
%SMOOTH_PART 计算光滑部分 f(x) = beta * sum log(1+exp(-b_j*a_j'*x)) + lambda2/2 * ||x||^2
% 输入:
%   x       - 决策变量（n x 1）
%   ba      - 标签加权数据矩阵（m x n）
%   beta    - 逻辑损失归一化系数
%   lambda2 - L2 正则化系数
% 输出:
%   val     - f(x) 的函数值
ax = ba * x;                        % ax(j) = b_j * a_j^T * x
val = beta * sum(stable_log1pexp(-ax)) + 0.5 * lambda2 * (x' * x);
end

function g = smooth_grad(x, ba, beta, lambda2)
%SMOOTH_GRAD 计算光滑部分的梯度。
%   grad f(x) = -beta * ba^T * sigma(-ba*x) + lambda2 * x
%   其中 sigma(z) = 1/(1+exp(z)) 为 sigmoid 函数的互补形式。
% 输入:
%   x       - 决策变量（n x 1）
%   ba      - 标签加权数据矩阵（m x n）
%   beta    - 逻辑损失归一化系数
%   lambda2 - L2 正则化系数
% 输出:
%   g       - 梯度向量（n x 1）
ax = ba * x;                        % ax(j) = b_j * a_j^T * x
w = inv_one_plus_exp(ax);           % w_i = 1 / (1 + exp(ax_i))，数值稳定
g = -beta * (ba' * w) + lambda2 * x;
end

function y = soft_threshold(x, tau)
%SOFT_THRESHOLD L1 范数的近端算子（逐分量软阈值）。
%   prox_{tau*||·||_1}(x)_i = sign(x_i) * max(|x_i| - tau, 0)
% 输入:
%   x   - 输入向量
%   tau - 阈值参数（步长 s 与正则系数的乘积）
% 输出:
%   y   - 软阈值后的向量
y = sign(x) .* max(abs(x) - tau, 0);
end

function y = stable_log1pexp(z)
%STABLE_LOG1PEXP 数值稳定的 log(1 + exp(z)) 计算。
%   利用恒等式：
%     log(1+exp(z)) = max(z,0) + log(1 + exp(-|z|))
%   避免大 z 值时的指数溢出。
% 输入/输出均为逐元素运算。
y = max(z, 0) + log1p(exp(-abs(z)));
end

function y = inv_one_plus_exp(z)
%INV_ONE_PLUS_EXP 数值稳定的 1 / (1 + exp(z)) 计算（即 sigmoid(-z)）。
%   对 z >= 0 和 z < 0 分别处理，避免 exp 的溢出：
%     z >= 0:  exp(-z) / (1 + exp(-z))
%     z <  0:  1 / (1 + exp(z))
y = zeros(size(z));
pos = z >= 0;
neg = ~pos;

y(pos) = exp(-z(pos)) ./ (1 + exp(-z(pos)));
ez = exp(z(neg));
y(neg) = 1 ./ (1 + ez);
end
