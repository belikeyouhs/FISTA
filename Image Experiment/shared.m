classdef shared
    % shared  共享工具库（静态方法集合）

    methods (Static)
        %% 1. Haar 小波变换（基于 Wavelet Toolbox）

        % 多级2D Haar小波正变换（返回 struct）
        function coeffs = wdec_fwd(img, levels)
            if nargin < 2 || isempty(levels), levels = 3; end
            [C, S] = wavedec2(img, levels, 'haar');
            coeffs = struct('c', {C}, 'S', {S}, 'wname', 'haar');
        end

        % 多级2D Haar小波逆变换
        function img = wdec_inv(coeffs)
            img = waverec2(coeffs.c, coeffs.S, coeffs.wname);
        end

        % 提取系数数值向量（struct 或 numeric 兼容）
        function c = wdec_coeffs(coeffs)
            if isstruct(coeffs), c = coeffs.c; else, c = coeffs(:); end
        end

        % 结构体感知的软阈值
        function out = wdec_soft_threshold(coeffs, alpha)
            c = shared.wdec_coeffs(coeffs);
            c = shared.soft_threshold(c, alpha);
            if isstruct(coeffs), out = coeffs; out.c = c; else, out = c; end
        end

        % 系数空间线性组合 z = a*x + b*y
        function z = coeff_axpby(a, x, b, y)
            xc = shared.wdec_coeffs(x);
            yc = shared.wdec_coeffs(y);
            zc = a * xc + b * yc;
            if isstruct(x), z = x; z.c = zc; else, z = zc; end
        end

        %% 2. 模糊算子与梯度

        % FFT循环卷积模糊
        function result = blur(img, kernel)
            [h, w] = size(img);
            K = shared.psf2otf_custom(kernel, [h, w]);
            result = real(ifft2(fft2(img) .* K));
        end

        % 模糊算子的转置（核翻转180度）
        function result = blur_T(img, kernel)
            result = shared.blur(img, kernel(end:-1:1, end:-1:1));
        end

        % PSF转OTF：将核居中填充到图像尺寸后做FFT
        function K = psf2otf_custom(psf, outSize)
            [kh, kw] = size(psf);
            [h, w] = deal(outSize(1), outSize(2));
            big = zeros(h, w);
            kh2 = floor(kh/2); kw2 = floor(kw/2);
            % 核中心对齐到 (1,1)，循环环绕到图像边界
            for i = 1:kh
                for j = 1:kw
                    ii = mod(i - kh2 - 1, h) + 1;
                    jj = mod(j - kw2 - 1, w) + 1;
                    big(ii, jj) = big(ii, jj) + psf(i, j);
                end
            end
            K = fft2(big);
        end

        % 正向算子 A = blur ∘ wdec_inv（小波域到图像域）
        function y = A_op(x_coeff, kernel, levels)
            y = shared.blur(shared.wdec_inv(x_coeff), kernel);
        end

        % 伴随算子 AT = wdec_fwd ∘ blur_T（图像域到小波域）
        function x = AT_op(img, kernel, levels)
            x = shared.wdec_fwd(shared.blur_T(img, kernel), levels);
        end

        % 目标函数梯度 g = 2 * AT(Ax - b)
        function g = gradf(x_coeff, b, kernel, levels)
            residual = shared.A_op(x_coeff, kernel, levels) - b;
            g = shared.AT_op(residual, kernel, levels);
            g.c = 2.0 * g.c;
        end

        % 计算Lipschitz常数 L = 2 * max|K|^2
        function L = lipschitz(kernel, shape)
            K = fft2(kernel, shape(1), shape(2));
            L = 2.0 * max(abs(K(:)).^2);
        end

        %% 3. 近端算子

        % 软阈值算子（L1近端算子）
        function y = soft_threshold(x, alpha)
            y = sign(x) .* max(abs(x) - alpha, 0);
        end

        %% 4. 目标函数

        % F(x) = ||Ax - b||^2 + λ||x||_1
        function val = F_obj(x_coeff, b, kernel, lam, levels)
            res = shared.A_op(x_coeff, kernel, levels) - b;
            c = shared.wdec_coeffs(x_coeff);
            val = sum(res(:).^2) + lam * sum(abs(c));
        end

        %% 5. 模糊核与退化

        % 生成高斯模糊核
        function k = gaussian_kernel(sz, sigma)
            if mod(sz, 2) == 0, sz = sz + 1; end
            half = floor(sz / 2);
            [xx, yy] = meshgrid(-half:half, -half:half);
            k = exp(-(xx.^2 + yy.^2) / (2 * sigma^2));
            k = k / sum(k(:));
        end

        % 生成运动模糊核
        function k = motion_kernel(len, angle_deg, sz)
            if nargin < 1 || isempty(len), len = 15; end
            if nargin < 2 || isempty(angle_deg), angle_deg = 0; end
            if nargin < 3 || isempty(sz)
                sz = max(3, 2 * ceil(len / 2) + 1);
            end
            if mod(sz, 2) == 0, sz = sz + 1; end

            half = floor(sz / 2);
            [xx, yy] = meshgrid(-half:half, -half:half);
            theta = angle_deg * pi / 180;
            dir_x = cos(theta);
            dir_y = -sin(theta);

            proj = xx * dir_x + yy * dir_y;
            dist = abs(-xx * dir_y + yy * dir_x);
            k = max(1 - dist, 0) .* (abs(proj) <= len / 2);

            if sum(k(:)) == 0
                k(half + 1, half + 1) = 1;
            end
            k = k / sum(k(:));
        end

        % 图像退化：模糊 + 加性高斯噪声
        function b = degrade(clean, kernel, noise_std, seed)
            if nargin < 3, noise_std = 1e-3; end
            if nargin < 4, seed = 0; end
            b = shared.blur(clean, kernel);
            if noise_std > 0
                rng(seed);
                b = b + noise_std * randn(size(b));
            end
            b = max(min(b, 1), 0);
        end

        %% 6. 图像质量指标

        % 计算 PSNR
        function val = compute_psnr(ref, test)
            mse = mean((ref(:) - test(:)).^2);
            if mse < eps
                val = Inf;
            else
                val = 10 * log10(1.0 / mse);
            end
        end

        % 计算 SSIM（11×11 高斯窗）
        function val = compute_ssim(ref, test)
            K1 = 0.01; K2 = 0.03; L = 1.0;
            C1 = (K1 * L)^2; C2 = (K2 * L)^2;
            win = fspecial('gaussian', 11, 1.5);
            win = win / sum(win(:));
            % 局部均值与方差
            mu1 = imfilter(ref, win, 'replicate');
            mu2 = imfilter(test, win, 'replicate');
            mu1_sq = mu1.^2;  mu2_sq = mu2.^2;
            mu1_mu2 = mu1 .* mu2;
            sigma1_sq = imfilter(ref.^2, win, 'replicate') - mu1_sq;
            sigma2_sq = imfilter(test.^2, win, 'replicate') - mu2_sq;
            sigma12 = imfilter(ref .* test, win, 'replicate') - mu1_mu2;
            % SSIM 公式
            ssim_map = ((2*mu1_mu2 + C1) .* (2*sigma12 + C2)) ./ ...
                       ((mu1_sq + mu2_sq + C1) .* (sigma1_sq + sigma2_sq + C2));
            val = mean(ssim_map(:));
        end
    end
end
