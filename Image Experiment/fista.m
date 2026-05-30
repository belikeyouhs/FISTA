function [img, result] = fista(b, kernel, lam, max_iter, tol, snap_iters, levels, target_f, verbose)
% FISTA  快速迭代收缩阈值算法（Fast Iterative Shrinkage-Thresholding Algorithm）
%
%   收敛速度 O(1/k²)：F(x_k) - F* ≤ 2L·||x0 - x*||² / (k+1)²
%
%   迭代公式:
%     x_k     = T_{λ/L}( y_k - (1/L)·∇f(y_k) )
%     t_{k+1} = (1 + √(1 + 4t_k²)) / 2
%     y_{k+1} = x_k + (t_k-1)/t_{k+1} · (x_k - x_{k-1})

    % 默认参数
    if nargin < 3  || isempty(lam),        lam = 2e-5; end
    if nargin < 4  || isempty(max_iter),   max_iter = 200; end
    if nargin < 5  || isempty(tol),        tol = 1e-6; end
    if nargin < 6  || isempty(snap_iters), snap_iters = [100, 200]; end
    if nargin < 7  || isempty(levels),     levels = 3; end
    if nargin < 8  || isempty(target_f),   target_f = Inf; end
    if nargin < 9  || isempty(verbose),    verbose = true; end

    L = shared.lipschitz(kernel, size(b));
    step = 1.0 / L;
    if verbose
        fprintf('  [FISTA]  L=%.6f  step=%.6e  lam=%g  wavelet=haar\n', L, step, lam);
        if isfinite(target_f)
            fprintf('           target_f=%.3e\n', target_f);
        end
    end

    x = shared.wdec_fwd(b, levels);  % x0 = W·b
    y = x;     % 辅助变量（动量外推点）
    t = 1.0;   % 动量时间步
    losses = zeros(1, max_iter);
    times  = zeros(1, max_iter);
    snaps  = struct();
    iter_reached = [];
    t0 = tic;

    for k = 1:max_iter
        x_prev = x;
        g = shared.gradf(y, b, kernel, levels);
        x = shared.coeff_axpby(1, y, -step, g);
        x = shared.wdec_soft_threshold(x, lam * step);

        loss = shared.F_obj(x, b, kernel, lam, levels);
        losses(k) = loss;
        times(k)  = toc(t0);

        if ismember(k, snap_iters) || mod(k, 100) == 0
            snaps.(['iter' num2str(k)]) = max(min(shared.wdec_inv(x), 1), 0);
        end

        % Nesterov 加速：t_{k+1} = (1+√(1+4t²))/2
        t_new = (1.0 + sqrt(1.0 + 4.0 * t^2)) / 2.0;
        y = shared.coeff_axpby(1, x, (t - 1.0) / t_new, shared.coeff_axpby(1, x, -1, x_prev));  % y = x + β(x - x_prev)
        t = t_new;

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

    if k < max_iter
        losses = losses(1:k);
        times  = times(1:k);
    end

    img = max(min(shared.wdec_inv(x), 1), 0);
    result = struct('losses', losses, 'times', times, 'snaps', snaps, ...
                    'x_final', x, 'iter_reached', iter_reached);
end
