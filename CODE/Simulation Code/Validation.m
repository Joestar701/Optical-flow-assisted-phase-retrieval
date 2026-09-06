clear; clc
close all;

%% Initialize File Paths
imageFolderPath = 'sphere_SIM_40dB\'; % Path to the interferogram image folder
t = length(dir(strcat(imageFolderPath,'*.bmp')));

% Initialize an array to store all interferogram images
images = zeros(512,512,t); % t interferogram images

for i = 1:t
    imagePath = fullfile(imageFolderPath, sprintf('%03d.bmp', i));
    currentImage = imread(imagePath);
    images(:,:,i) = currentImage;
end

% figure; imshow(images(:,:,150),[]);
% figure; plot(squeeze(images(232,271,:)));

%% Compare Simulated and Predicted Phase-Shift Errors
data = load('infphase.txt');
vibPre = data(:,2);

yyaxis left
plot(vibPre-pi/2, 'Color', [223/255 124/255 113/255], 'LineWidth', 1);

vibPhase = load('vibPhase.txt');

% Calculate the phase-shift difference between adjacent frames
% For example: 132-131, 133-132, ...
dphi = diff(vibPhase);

hold on
plot(-dphi, '--', 'Color', [43/255 106/255 153/255], 'LineWidth', 1);
ylim([-1 1.5])

yyaxis right

% Calculate the difference between the simulated and predicted errors
diff = vibPre-pi/2+dphi;

% writematrix(diff, 'diff3.txt');

meadiff = mean(diff);
stddiff = std(diff);

plot(diff,'Color', [41/255 134/255 204/255],'LineWidth', 1);
ylim([-0.2 1])

legend({'Simulated phase-shift error', ...
        'Predicted phase-shift error', ...
        'Difference between the two errors'}, ...
        'Location', 'northwest');

ax = gca;
ax.YAxis(1).Color = [223/255 124/255 113/255]; % Match the simulated phase-shift error
ax.YAxis(2).Color = [41/255 134/255 204/255];

% figure,plot(vibPhase, 'Color', [51/255 134/255 204/255], 'LineWidth', 0.7);

%% Phase Reconstruction
phi = [0; cumsum(vibPre)];

[H, W, Z] = size(images);

% Construct the phase-shifting model matrix
M = [ones(Z,1), cos(phi), sin(phi)];

tic

% Compute the pseudoinverse of the phase-shifting model matrix
Minv = (M' * M) \ M';                  % 3 x Z

% Reshape the image sequence into a matrix
I_all = reshape(double(images), H*W, Z)';   % Z x (H*W)

% Estimate the phase-shifting coefficients
p_all = Minv * I_all;                  % 3 x (H*W)

% Extract the wrapped phase
Phase_vec = atan2(-p_all(3,:), p_all(2,:));

% Reshape the phase vector into a 2D phase map
Phase = reshape(Phase_vec, H, W);

toc

times = toc/512/512;

% Perform 2D phase unwrapping
phase = unwrap_WLS_FFT(Phase);

% figure, mesh(phase);

% Convert phase to surface height
h = phase * 455 / (4*pi);

% figure, mesh(h);

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

%% Remove the Reference Surface
TG = h - Zref;

figure, mesh(-TG);
axis off
