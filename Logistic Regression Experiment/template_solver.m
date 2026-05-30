function result = template_solver(problem, opts)
%TEMPLATE_SOLVER 求解器模板，用于实现新算法。
%
% 基于复合优化问题 min_x f(x) + h(x) 的近端梯度框架：
%   每步迭代执行一次近端梯度步：
%     x_{k+1} = prox_{s*h}(x_k - s * grad f(x_k))
%   其中 s = 1/L 为步长，L 为 Lipschitz 常数上界。
%
% 使用方法：
%   将循环体中的更新规则替换为自定义算法步骤即可。
%
% 输入:
%   problem - 问题结构体，包含：
%     .x0       - 初始点（n x 1）
%     .F        - 目标函数 F(x) = f(x) + h(x)
%     .grad     - 光滑部分梯度 grad f(x)
%     .prox_h   - 非光滑部分近端算子 prox_{s*h}(x)
%     .L_upper  - 光滑部分 Lipschitz 常数上界
%     .exit_crit - 停止准则句柄
%   opts    - 选项结构体，包含：
%     .name     - 算法名称
%     .maxIter  - 最大迭代次数
%     .tol      - 停止阈值
%
% 输出:
%   result  - 结构体，包含：
%     .x        - 最终迭代点
%     .cost     - 每步目标函数值序列
%     .time     - 每步累计耗时序列
%     .iter     - 总迭代次数

opts = default_solver_opts(opts);       % 填充默认选项

x = problem.x0;                         % 当前迭代点
cost = problem.F(x);                   % 初始目标函数值
time = 0;                               % 累计耗时
tStart = tic;

for k = 1:opts.maxIter
    x_prev = x;                         % 保存前一步迭代点

    % TODO: 将以下占位符替换为自定义算法步骤。
    % 当前为标准近端梯度步（ISTA）：
    %   x = prox_{s*h}(x - s * grad f(x)),  s = 1/L
    x = problem.prox_h(x - (1 / problem.L_upper) * problem.grad(x), 1 / problem.L_upper);

    cost(end+1, 1) = problem.F(x); %#ok<AGROW>
    time(end+1, 1) = toc(tStart); %#ok<AGROW>

    if problem.exit_crit(x, x_prev) < opts.tol
        break;                          % 满足停止条件，退出
    end
end

result = struct();
result.name = opts.name;                % 算法名称
result.x = x;                           % 最终迭代点
result.cost = cost;                     % 目标函数值序列
result.time = time;                     % 耗时序列
result.restartIters = [];              % 重启迭代编号（无重启则为空）
result.Lhist = [];                     % Lipschitz 常数估计序列
result.btIters = [];                   % 回溯迭代次数序列
result.iter = numel(cost) - 1;         % 总迭代次数
end
