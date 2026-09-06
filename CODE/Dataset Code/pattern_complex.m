function [pattern, meta] = pattern_complex(imSize, opts)
%PATTERN_COMPLEX Generate a continuous complex surface pattern with local spots.
%
%   This function generates physically consistent surface patterns for
%   optical-flow-based phase-shift estimation. Two pattern types are
%   supported:
%
%   - 'surface+spot'
%   - 'curvature+spot'
%
%   Syntax:
%       [pattern, meta] = pattern_complex([H,W], opts)
%
%   Input:
%       imSize          Image size [H, W].
%
%       opts.type       Pattern type:
%                       'surface+spot' | 'curvature+spot'
%
%       opts.spotNum    Number of local spots.
%
%       opts.spotSigma  Gaussian sigma of the local spots (pixels).
%
%       opts.surfaceSigma
%                       Gaussian smoothing sigma for the random surface.
%
%       opts.curvature  Scaling factor for the spherical surface radius.
%
%       opts.amplitude  Amplitude scaling factor for the surface
%                       height/phase.
%
%       opts.normalize  Normalization mode:
%                       'none' | '01' | 'phase'
%
%       opts.seed       Random seed for reproducible pattern generation.
%
%   Output:
%       pattern         Generated surface/phase pattern.
%
%       meta            Metadata associated with the generated pattern,
%                       including the spot centers and pattern type.
%
% -------------------------------------------------------------------------

    arguments
        imSize (1,2) {mustBePositive, mustBeInteger}
        opts.type char {mustBeMember(opts.type, ...
            {'surface+spot','curvature+spot'})} ...
            = 'curvature+spot'
        opts.spotNum (1,1) double = 5
        opts.spotSigma (1,1) double = 5
        opts.surfaceSigma (1,1) double = 15
        opts.curvature double = 1.2
        opts.amplitude double = 10.0
        opts.normalize char {mustBeMember(opts.normalize, ...
            {'none','01','phase'})} ...
            = 'phase'
        opts.seed double = []
    end

    H = imSize(1);
    W = imSize(2);

    if ~isempty(opts.seed)
        rng(opts.seed);
    end

    %% --------------------------------------------------------------------
    % Generate Base Surface Pattern
    % ---------------------------------------------------------------------
    switch opts.type

        case 'surface+spot'
            % Generate a smooth random continuous surface.
            noise = 5 * randn(H, W);
            base = imgaussfilt(noise, opts.surfaceSigma);

            base = base - min(base(:));
            base = base / max(base(:));
            base = base * opts.amplitude;

        case 'curvature+spot'
            % Generate a continuous spherical/curved surface.
            [XX, YY] = meshgrid(1:W, 1:H);

            % Select the sphere center along the x-axis.
            % Discrete sampling is used to avoid a completely symmetric
            % surface configuration.
            xposOptions = round(linspace(1, W, 9));
            cx = xposOptions(randi(numel(xposOptions)));
            cy = (H + 1) / 2;

            % Define the spherical radius.
            R0 = sqrt((W/2)^2 + (H/2)^2);
            R  = R0 * opts.curvature;

            dx = min(abs(XX - cx), W - abs(XX - cx));
            dy = YY - cy;
            dist2 = dx.^2 + dy.^2;

            mask = dist2 <= R^2;

            Z = zeros(H, W);
            Z(mask) = sqrt(R^2 - dist2(mask));

            base = Z;
            base = base - min(base(:));
            base = base / max(base(:));
            base = base * opts.amplitude;

    end

    %% --------------------------------------------------------------------
    % Add Local Spots
    % ---------------------------------------------------------------------
    pattern = base;
    meta.spotCenters = zeros(opts.spotNum, 2);

    for k = 1:opts.spotNum

        % Randomly select the center of each local spot.
        cx_spot = randi([1 W]);
        cy_spot = randi([1 H]);

        meta.spotCenters(k,:) = [cy_spot, cx_spot];

        % Generate a Gaussian spot.
        spot = fspecial('gaussian', [H W], opts.spotSigma);

        spot = circshift( ...
            spot, ...
            [cy_spot - round(H/2), ...
             cx_spot - round(W/2)]);

        spot = spot / max(spot(:));

        % Randomly vary the amplitude of each spot.
        pattern = pattern + spot * (0.5 + 0.5 * rand());

    end

    %% --------------------------------------------------------------------
    % Normalization and Phase Mapping
    % ---------------------------------------------------------------------
    switch opts.normalize

        case '01'
            % Normalize the pattern to the range [0, 1].
            pattern = normalize01(pattern);

        case 'phase'
            % Map the normalized surface pattern to phase.
            pattern = 2*pi * pattern;

        case 'none'
            % No normalization or phase mapping is applied.

    end

    meta.type = opts.type;

end

%% ========================================================================
% Utility Function
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
