%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Synthetic Interferogram-Pair Dataset Generation for Unequal
% Phase-Shift Estimation in Phase-shifting Interferometric Measurement
%
% This code generates synthetic interferogram pairs with unequal
% phase shifts for training a RAFT-based phase-shift estimation network.
% 
% The generated data consist of:
%   * *_pic1.bmp   : First interferogram
%
%   * *_pic2.bmp   : Second interferogram with phase shift
%
%   * *_params.txt : Ground-truth phase shift and simulation parameters

% Author: Qizhe Wu (Huazhong University of Science and Technology)
% Email: qizhewu@hust.edu.cn
% Date: 2026-09-06

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ========================================================================
% Initialize workspace
% =========================================================================
clear;
clc;
close all;

%% Basic Parameters
lambda = 455;              % Wavelength (nm)
lc = 11501;                % Coherence length (nm)
Nx = 512;                  % Number of pixels along the x-axis
Ny = 512;                  % Number of pixels along the y-axis
FOV = 3577000;             % Field of view width (nm)
Pixelsize = FOV/(Ny-1);    % Pixel size in the simulation plane (nm)
M = 27.06;                 % System magnification

% Output directory
outDir = 'TrainData_BMP';

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% Surface Pattern Types
% Five different surface-pattern types are used to increase the diversity
% of the synthetic training dataset.
patternCases = { ...
    'generate_phase_map', ...
    'pattern_shapes', ...
    'pattern_lr_step', ...
    'surface+spot', ...
    'curvature+spot' ...
    };

numPerCase = 1200;
totalSamples = numel(patternCases) * numPerCase;    % 6000 samples

%% Generate Synthetic Dataset
for i = 1:totalSamples

    fprintf('Generating sample %d / %d...\n', i, totalSamples);

    %% --------------------------------------------------------------------
    % Random Fringe Angle and Orientation
    % ---------------------------------------------------------------------
    % alpha controls the fringe density, while theta determines the
    % orientation of the interference fringes.

    alpha = (pi/3000) + ...
        (pi/600 - pi/3000) * rand;

    theta = 2*pi*rand;     % Fringe orientation angle [0, 2*pi)

    K = 4*pi*sin(alpha)/lambda;

    Kx = K * cos(theta);
    Ky = K * sin(theta);

    [xg, yg] = meshgrid( ...
        ((1:Nx)-(Nx+1)/2) * Pixelsize/M, ...
        ((1:Ny)-(Ny+1)/2) * Pixelsize/M);

    % Reference-wave phase distribution
    Pref = Kx * xg + Ky * yg;

    % Reference surface height
    Zref = Pref * lambda / (4*pi);

    %% --------------------------------------------------------------------
    % Select Surface Pattern
    % ---------------------------------------------------------------------
    caseIdx = ceil(i/numPerCase);
    caseIdx = min(caseIdx, numel(patternCases));
    ptype = patternCases{caseIdx};

    %% --------------------------------------------------------------------
    % Generate Surface Phase Distribution
    % ---------------------------------------------------------------------
    switch ptype

        case 'generate_phase_map'
            spot = generate_phase_map([Ny, Nx]);

        case 'pattern_lr_step'
            [pat, ~] = pattern_lr_step([Ny, Nx]);
            spot = generate_phase_map([Ny, Nx]) + pat;

        case 'pattern_shapes'
            [pat, ~, ~] = pattern_shapes([Ny, Nx]);
            spot = generate_phase_map([Ny, Nx]) + pat;

        otherwise
            spot = pattern_complex([Ny, Nx], 'type', ptype) ...
                   + generate_phase_map([Ny, Nx]);

    end

    %% --------------------------------------------------------------------
    % Convert Surface Phase to Surface Height
    % ---------------------------------------------------------------------
    Zobj = spot * lambda / (4*pi);

    %% --------------------------------------------------------------------
    % Generate Non-Uniform Phase Shift
    % ---------------------------------------------------------------------
    % vibphase represents the phase shift between the two consecutive
    % interferograms.

    vibphase = 2 * (rand() - 0.5) + 0.8;

    %% --------------------------------------------------------------------
    % Generate Interferograms
    % ---------------------------------------------------------------------
    % Gaussian envelope determined by the optical path difference.

    Gauss = exp(-4*pi^2 * ((Zobj - Zref)/lc).^2);

    % First interferogram
    pic1 = 125 * cos(4*pi*(Zobj - Zref)/lambda) + 125;

    % Second interferogram with a non-uniform phase shift
    pic2 = 125 * cos( ...
        4*pi*(Zobj - Zref)/lambda - vibphase) + 125;

    %% --------------------------------------------------------------------
    % Add Gaussian Noise
    % ---------------------------------------------------------------------
    noisevalue = 0.0003 * rand();

    pic1_noisy = imnoise( ...
        uint8(pic1), 'gaussian', 0, noisevalue);

    pic2_noisy = imnoise( ...
        uint8(pic2), 'gaussian', 0, noisevalue);

    %% --------------------------------------------------------------------
    % Save Interferogram Images
    % ---------------------------------------------------------------------
    base_filename = sprintf('sample_%04d', i);

    imwrite( ...
        pic1_noisy, ...
        fullfile(outDir, [base_filename '_pic1.bmp']));

    imwrite( ...
        pic2_noisy, ...
        fullfile(outDir, [base_filename '_pic2.bmp']));

    %% --------------------------------------------------------------------
    % Save Ground-Truth Parameters
    % ---------------------------------------------------------------------
    param_filename = fullfile( ...
        outDir, [base_filename '_params.txt']);

    fid = fopen(param_filename, 'w');

    if fid ~= -1

        fprintf(fid, 'vibphase=%.6f\n', vibphase);
        fprintf(fid, 'alpha=%.6f\n', alpha);
        fprintf(fid, 'pattern_type=%s\n', ptype);
        fprintf(fid, 'noise_level=%.6f\n', noisevalue);

        fclose(fid);

    end

    %% --------------------------------------------------------------------
    % Display Progress
    % ---------------------------------------------------------------------
    if mod(i, 500) == 0
        fprintf( ...
            'Completed %d / %d samples.\n', ...
            i, totalSamples);
    end

end

fprintf('Synthetic training dataset generation completed successfully.\n');

