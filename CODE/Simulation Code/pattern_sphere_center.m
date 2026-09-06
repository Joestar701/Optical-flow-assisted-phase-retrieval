function [pattern, meta] = pattern_sphere_center(imSize, opts)
%PATTERN_SPHERE_CENTER Generate a spherical phase pattern centered in the image.
%
%   [pattern, meta] = pattern_sphere_center([H,W], opts)
%
%   Input:
%       imSize          Image size [H, W].
%
%   Optional inputs:
%       opts.curvature  Scaling factor for the spherical radius.
%       opts.amplitude  Amplitude of the spherical phase pattern.
%       opts.spotNum    Number of local spots.
%       opts.spotSigma  Gaussian sigma of the local spots (pixels).
%       opts.normalize  Output normalization mode:
%                       'none' | '01' | 'phase'
%       opts.seed       Random seed for reproducible pattern generation.
%
%   Output:
%       pattern         Generated spherical phase pattern.
%       meta            Metadata containing the pattern parameters and
%                       locations of the added local spots.

    arguments
        imSize (1,2) {mustBePositive, mustBeInteger}
        opts.curvature double = 1.2
        opts.amplitude double = 5.0
        opts.spotNum (1,1) double = 2
        opts.spotSigma (1,1) double = 5
        opts.normalize char {mustBeMember(opts.normalize,{'none','01','phase'})} = 'phase'
        opts.seed double = []
    end

    H = imSize(1);
    W = imSize(2);

    if ~isempty(opts.seed)
        rng(opts.seed);
    end

    %% -------------------- Generate Centered Spherical Surface --------------------
    [XX,YY] = meshgrid(1:W,1:H);

    % Image center
    cx = (W + 1)/2;
    cy = (H + 1)/2;

    % Base radius
    R0 = sqrt((W/2)^2 + (H/2)^2);
    R  = R0 * opts.curvature;

    dx = XX - cx;
    dy = YY - cy;
    dist2 = dx.^2 + dy.^2;

    % Generate the spherical surface
    mask = dist2 <= R^2;
    Z = zeros(H,W);
    Z(mask) = sqrt(R^2 - dist2(mask));

    % Normalize the spherical surface
    Z = Z - min(Z(:));
    Z = Z / max(Z(:));
    Z = Z * opts.amplitude;

    pattern = Z;

    %% -------------------- Add Local Spots --------------------
    meta.spotCenters = zeros(opts.spotNum, 2);

    for k = 1:opts.spotNum
        cx_spot = randi([1 W]);
        cy_spot = randi([1 H]);
        meta.spotCenters(k,:) = [cy_spot, cx_spot];

        spot = fspecial('gaussian', [H W], opts.spotSigma);
        spot = circshift(spot, [cy_spot - round(H/2), cx_spot - round(W/2)]);
        spot = spot / max(spot(:));

        pattern = pattern + spot * (0.5 + 0.5 * rand());
    end

    %% -------------------- Normalize the Output --------------------
    switch opts.normalize
        case '01'
            pattern = normalize01(pattern);

        case 'phase'
            pattern = 2*pi * pattern;

        case 'none'
    end

    meta.type = 'sphere_center';
    meta.center = [cy, cx];
    meta.radius = R;
end


%% -------------------- Utility Function --------------------
function A = normalize01(A)
    mn = min(A(:));
    mx = max(A(:));

    if mx == mn
        A = zeros(size(A));
    else
        A = (A - mn) / (mx - mn);
    end
end

