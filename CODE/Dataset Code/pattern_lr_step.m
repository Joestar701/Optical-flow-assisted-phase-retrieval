function [pattern, meta] = pattern_lr_step(imSize, opts)
%PATTERN_LR_STEP Generate a left-to-right stepwise phase map.
%
%   The phase is uniform along the y-direction, while 1 to N phase
%   discontinuities are introduced along the x-direction.
%
%   [pattern, meta] = pattern_lr_step(imSize, opts)
%
%   Optional inputs:
%       opts.seed       Random seed for reproducible generation.
%       opts.addNoise   Add local Airy-like perturbations, default true.
%       opts.numSteps   Number of step regions, default 2 to 4.
%       opts.stepPos    Manually specified step positions as ratios in [0,1].
%
%   Outputs:
%       pattern         Generated stepwise phase map.
%       meta            Metadata containing the step parameters.

arguments
    imSize (1,2) double = [512,512]
    opts.seed double = []
    opts.addNoise logical = true
    opts.numSteps double = []
    opts.stepPos  double = []
end

%% ========================================================================
% Set Random Seed
% =========================================================================
if ~isempty(opts.seed)
    rng(opts.seed);
end

H = imSize(1);
W = imSize(2);

pattern = zeros(H, W);

%% ========================================================================
% 1) Determine the Number of Steps
% =========================================================================
if isempty(opts.numSteps)

    % Randomly select 2 to 4 step regions.
    numSteps = randi([2,4]);

else

    numSteps = opts.numSteps;

end

%% ========================================================================
% 2) Determine Step Positions Along the x-Axis
% =========================================================================
if isempty(opts.stepPos)

    % Generate numSteps-1 internal boundaries as normalized positions.
    cuts = sort(0.15 + 0.7*rand(1, numSteps-1));

else

    cuts = sort(opts.stepPos(:)');

end

xCuts = round([0, cuts, 1] * W);

xCuts(1) = 1;
xCuts(end) = W;

%% ========================================================================
% 3) Generate Phase Values for Each Step
% =========================================================================
% The phase values are uniformly distributed within [-2, 2] rad.
phaseVals = -2 + 4*rand(1, numSteps);

%% ========================================================================
% 4) Construct the Stepwise Phase Map
% =========================================================================
for i = 1:numSteps

    x0 = xCuts(i);
    x1 = xCuts(i+1);

    pattern(:, x0:x1) = phaseVals(i);

end

%% ========================================================================
% 5) Add Optional Airy-Like Perturbations
% =========================================================================
if opts.addNoise

    spotNum = randi([5,12]);

    airy = Airy_map(imSize, spotNum);

    pattern = pattern + ...
        (0.05 + 0.25*rand()) * airy;

end

%% ========================================================================
% Store Metadata
% =========================================================================
meta.numSteps  = numSteps;
meta.stepPos   = cuts;
meta.phaseVals = phaseVals;

end


%% ========================================================================
% Generate Airy-Like Phase Perturbations
% =========================================================================
function phaseMap = Airy_map(imSize, numCylinders)

    plane = zeros(imSize);

    [X, Y] = meshgrid(1:imSize(2), 1:imSize(1));

    for i = 1:numCylinders

        cx = randi([1, imSize(2)]);
        cy = randi([1, imSize(1)]);

        radius = randi([1, 3]);
        height = 1.5 + 0.5*rand();

        mask = (sqrt((X-cx).^2 + (Y-cy).^2) <= radius);

        plane(mask) = height;

    end

    psfKernel = airy_PSF(256);

    blurred = imfilter( ...
        plane, psfKernel, 'replicate', 'conv');

    phaseMap = 2*pi*normalize01(blurred);

end


%% ========================================================================
% Generate Airy PSF
% =========================================================================
function psf = airy_PSF(m)

    raDiusOut = randi([16, 64]);
    raDiusIn = 0;

    mapSpace = ones(m);
    r = fix(m/2);

    for x = -r+1:r

        for y = -r+1:r

            if x^2 + y^2 >= (raDiusOut^2)
                mapSpace(x+r, y+r) = 0;
            end

            if x^2 + y^2 < (raDiusIn^2)
                mapSpace(x+r, y+r) = 0;
            end

        end

    end

    mtf = conv2(mapSpace, mapSpace);

    mtf_fft = fftshift( ...
        fft2(fftshift(mtf)));

    psf = abs(mtf_fft).^2;

    % Normalize the PSF.
    psf = psf / max(psf(:));

end


%% ========================================================================
% Normalize an Array to [0, 1]
% =========================================================================
function A = normalize01(A)

    mn = min(A(:));
    mx = max(A(:));

    if mx == mn

        A = zeros(size(A));

    else

        A = (A - mn) / (mx - mn);

    end

end
