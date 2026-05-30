function [img, result] = ista(b, kernel, lam, max_iter, tol, snap_iters, levels, target_f, verbose)
% ISTA  迭代收缩阈值算法（Iterative Shrinkage-Thresholding Algorithm）
%
%   收敛速度 O(1/k)：F(x_k) - F* ≤ C/k
%
%   迭代公式:
%     x_{k+1} = T_{λ/L}( x_k - (1/L)·∇f(x_k) )
%   其中 T_α 为软阈值算子，L 为 Lipschitz 常数
%
%   输入: b=模糊图像, kernel=PSF, lam=正则化系数, tol=收敛阈值
%   输出: img=重建图像, result=结构体(losses, times, snaps, x_final, iter_reached)

    % 默认参数
    if nargin < 3 || isempty(lam),        lam = 2e-5; end
    if nargin < 4 || isempty(max_iter),   max_iter = 200; end
    if nargin < 5 || isempty(tol),        tol = 1e-6; end
    if nargin < 6 || isempty(snap_iters), snap_iters = [100, 200]; end
    if nargin < 7 || isempty(levels),     levels = 3; end
    if nargin < 8 || isempty(target_f),   target_f = Inf; end
    if nargin < 9 || isempty(verbose),    verbose = true; end

    L = shared.lipschitz(kernel, size(b));  % L = 2·max|K(ω)|²
    step = 1.0 / L;
    if verbose
        fprintf('  [ISTA]   L=%.6f  step=%.6e  lam=%g  wavelet=haar\n', L, step, lam);
        if isfinite(target_f)
            fprintf('           target_f=%.3e\n', target_f);
        end
    end

    % 初始化：x0 = W^T b
    x = shared.wdec_fwd(b, levels);
    losses = zeros(1, max_iter);
    times  = zeros(1, max_iter);
    snaps  = struct();
    iter_reached = [];
    t0 = tic;

    for k = 1:max_iter
        x_prev = x;
        g = shared.gradf(x, b, kernel, levels);  % g = 2·AT(Ax - b)
        x = shared.coeff_axpby(1, x, -step, g);         % 梯度下降
        x = shared.wdec_soft_threshold(x, lam * step);  % 近端投影

        loss = shared.F_obj(x, b, kernel, lam, levels);
        losses(k) = loss;
        times(k)  = toc(t0);

        if ismember(k, snap_iters) || mod(k, 100) == 0
            snaps.(['iter' num2str(k)]) = max(min(shared.wdec_inv(x), 1), 0);
        end

        % 相对变化量 ||x_k - x_{k-1}|| / ||x_{k-1}||
        rel = norm(shared.wdec_coeffs(x) - shared.wdec_coeffs(x_prev)) / (norm(shared.wdec_coeffs(x_prev)) + 1e-12);

        if verbose && (ismember(k, snap_iters) || k == 1 || mod(k, 100) == 0)
            fprintf('  iter %5d | F=%.4e | drel=%.2e\n', k, loss, rel);
        end

        if isfinite(target_f) && loss <= target_f
            iter_reached = k;
            if verbose, fprintf('  OK target reached F=%.4e <= %.3e at iter %d\n', loss, target_f, k); end
            break;
        end
        if rel < tol && k > 5
            if verbose, fprintf('  OK converged at iter %d\n', k); end
            break;
        end
    end

    % 截断未使用的预分配
    if k < max_iter
        losses = losses(1:k);
        times  = times(1:k);
    end

    img = max(min(shared.wdec_inv(x), 1), 0);
    result = struct('losses', losses, 'times', times, 'snaps', snaps, ...
                    'x_final', x, 'iter_reached', iter_reached);
end
