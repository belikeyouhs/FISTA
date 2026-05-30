%% blur_image.m — 生成模糊图像并保存为 blurred.mat

clc; clear; close all;

%% 用户参数
input_file  = '21077.jpg';   % 输入图像路径21077.jpg\37073.jpg\87065.jpg\108005.jpg
output_file = 'blurred.mat';     % 输出文件
noise_std   = 1e-3;              % 加性噪声标准差
levels      = 3;                 % 小波级数（用于尺寸裁剪）

% 模糊类型: 'gaussian' 或 'motion'
blur_type = 'gaussian';

% 高斯模糊参数
gaussian_kernel_size = 9;         % 高斯核尺寸（奇数）
sigma = 4.0;                      % 高斯核标准差

% 运动模糊参数
motion_length = 15;               % 运动轨迹长度（像素），越大模糊越强
motion_angle = 30;                % 运动方向角度（度），0=水平向右，90=竖直向上
motion_kernel_size = 17;          % 运动核尺寸（奇数）；[] 表示根据 length 自动设置
%% ═════════════════════════════════════════════════════════════

% 确保 shared.m 在路径中
if ~exist('shared', 'class')
    addpath(fileparts(mfilename('fullpath')));
end

% 读取并转灰度
if exist(input_file, 'file')
    img = imread(input_file);
else
    % 使用 MATLAB 内置图像作为备选
    fprintf('  文件 "%s" 不存在，使用内置 cameraman 图像\n', input_file);
    img = imread('cameraman.tif');
end

% 归一化至 [0,1]
if ndims(img) == 3
    clean = double(rgb2gray(img)) / 255.0;
else
    clean = double(img) / 255.0;
end
fprintf('  灰度图尺寸: %dx%d  值域: [%.3f, %.3f]\n', size(clean,1), size(clean,2), min(clean(:)), max(clean(:)));

% 裁剪至小波分解所需尺寸
d = 2^levels;
h = floor(size(clean,1) / d) * d;
w = floor(size(clean,2) / d) * d;
clean = clean(1:h, 1:w);
fprintf('  裁剪至 %d 的倍数: %dx%d\n', d, h, w);

% 生成模糊核并退化
switch lower(blur_type)
    case 'gaussian'
        kernel = shared.gaussian_kernel(gaussian_kernel_size, sigma);
        blur_params = struct('type', blur_type, ...
                             'kernel_size', gaussian_kernel_size, ...
                             'sigma', sigma);
        fprintf('  模糊类型: Gaussian, kernel_size=%d, sigma=%.3f\n', ...
                gaussian_kernel_size, sigma);

    case 'motion'
        kernel = shared.motion_kernel(motion_length, motion_angle, motion_kernel_size);
        blur_params = struct('type', blur_type, ...
                             'length', motion_length, ...
                             'angle', motion_angle, ...
                             'kernel_size', size(kernel, 1));
        fprintf('  模糊类型: Motion, length=%.2f, angle=%.2f°, kernel_size=%d\n', ...
                motion_length, motion_angle, size(kernel, 1));

    otherwise
        error('未知模糊类型 "%s"。可选值: gaussian, motion。', blur_type);
end

% 施加模糊+噪声
blurred = shared.degrade(clean, kernel, noise_std, 0);

% 评估退化图质量
psnr_blur = shared.compute_psnr(clean, blurred);
ssim_blur = shared.compute_ssim(clean, blurred);
fprintf('  模糊图 PSNR = %.2f dB, SSIM = %.4f\n', psnr_blur, ssim_blur);

% 保存到 .mat
save(output_file, 'clean', 'blurred', 'kernel', 'blur_type', 'blur_params', ...
                  'sigma', 'gaussian_kernel_size', ...
                  'motion_length', 'motion_angle', 'motion_kernel_size', ...
                  'noise_std', 'levels');
fprintf('  已保存至 %s\n', output_file);

% 可视化
figure('Name', '图像模糊化');
subplot(1,3,1); imshow(clean);   title('原始清晰图');
subplot(1,3,2); imshow(blurred); title(sprintf('模糊+噪声图 (PSNR=%.2fdB, SSIM=%.4f)', psnr_blur, ssim_blur));
subplot(1,3,3); imagesc(kernel); axis image off; colormap gray; title('模糊核');
