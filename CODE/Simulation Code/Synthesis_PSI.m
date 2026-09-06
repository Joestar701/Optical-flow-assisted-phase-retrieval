%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% This script generates synthetic VSI interferogram images with 
% unequal phase shifts and Gaussian noise for simulation and validation.
% 
% Author: Qizhe Wu (Huazhong University of Science and Technology)
% Email: qizhewu@hust.edu.cn
% Date: 2026-09-06

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% Simulation Parameters
lambda = 455;          % Wavelength (nm)
lc = 11501;            % Coherence length (nm)
Nx = 512; Ny = 512;
FOV = 4599000;         % Field of view (nm)
Pixelsize = FOV/(Ny-1);
alpha = pi/1000;
M = 27.06;
K = 4*pi*tan(alpha)/lambda;

dz = lambda/8;         % VSI scanning step (nm)
Nz = 30;               % Number of scanning frames

%% Tilted Reference Surface
x = ((1:Ny)-(Ny+1)/2) * Pixelsize/M;
y = ((1:Nx)-(Nx+1)/2) * Pixelsize/M;
[Y, X] = meshgrid(y, x);

theta = 30/180*pi;     % Fringe inclination angle
Kx = K * cos(theta);
Ky = K * sin(theta);

Pref = Kx * X + Ky * Y;
Zref = Pref * lambda / (4*pi);

%% Object Surface
pattern = pattern_sphere_center([Ny, Nx]);
Zobj = pattern * lambda / (4*pi);

figure, mesh(Zobj); axis off

%% Vibration-Induced Phase Shift
% vibPhase = rand(1,Nz)-0.5;
% writematrix(vibPhase.', 'vibPhase.txt');
vibPhase = load('vibPhase.txt');

figure;
plot(vibPhase);

%% Generate VSI Interferogram Sequence
z_scan = ((1:Nz) - (Nz+1)/2) * dz;  % Scan around the zero optical path difference
VSI = zeros(Ny, Nx, Nz, 'uint8');

SNR_dB = 40;           % Signal-to-noise ratio (dB)
V = 1;                 % Fringe contrast coefficient

for k = 1:Nz
    DeltaZ = Zobj - Zref - z_scan(k);

    % Gauss = exp(-4*pi^2 * (DeltaZ/lc).^2);

    pic = 125 * V * cos(4*pi*DeltaZ/lambda + 1*vibPhase(k)) + 125;

    pic_double = double(pic);

    % Estimate the signal power and determine the noise level
    signal_power = var(pic_double(:));
    noise_power = signal_power / (10^(SNR_dB/10));
    sigma = sqrt(noise_power);

    % Add Gaussian noise
    noise = sigma * randn(size(pic_double));
    noisy_img = pic_double + noise;

    % Clip pixel values to the valid 8-bit grayscale range
    noisy_img(noisy_img<0) = 0;
    noisy_img(noisy_img>255) = 255;

    VSI(:,:,k) = uint8(noisy_img);
end

figure;
imshow(VSI(:,:,10));

%% Save the Simulated Interferogram Sequence
outputFolder = 'Study\step_SIM_spot20';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

for i = 1:Nz
    filename = fullfile(outputFolder, sprintf('%03d.bmp', i));  % 001.bmp ~ 030.bmp
    imwrite(uint8(VSI(:,:,i)), filename);
end

