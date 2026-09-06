function [pattern, meta] = pattern_step_spot(imSize, opts)
%PATTERN_STEP_SPOT Generate a single step pattern with local spots.
%
%   [pattern, meta] = pattern_step_spot([H,W], opts)
%
%   Optional inputs:
%       opts.height      Step height (rad).
%       opts.position    Step position (0 to 1).
%       opts.direction   Step direction: 'x' | 'y'.
%       opts.spotNum     Number of local spots.
%       opts.spotSigma   Gaussian sigma of the local spots (pixels).
%       opts.seed        Random seed for reproducible pattern generation.
%
%   Outputs:
%       pattern          Generated stepwise phase pattern.
%       meta             Metadata containing the step parameters and
%                        locations of the added local spots.

    arguments
        imSize (1,2) {mustBePositive, mustBeInteger}
        opts.height double = 1
        opts.position double = 0.5
        opts.direction char {mustBeMember(opts.direction,{'x','y'})} = 'x'
        opts.spotNum (1,1) double = 20
        opts.spotSigma (1,1) double = 5
        opts.seed double = []
    end

    H = imSize(1);
    W = imSize(2);

    if ~isempty(opts.seed)
        rng(opts.seed);
    end

    %% -------------------- Generate Step Pattern --------------------

    pattern = zeros(H,W);

    switch opts.direction
        case 'x'
            step_col = round(W * opts.position);
            pattern(:, step_col:end) = opts.height;

        case 'y'
            step_row = round(H * opts.position);
            pattern(step_row:end,:) = opts.height;
    end

    %% -------------------- Add Local Spots --------------------

    meta.spotCenters = zeros(opts.spotNum,2);

    for k = 1:opts.spotNum

        cx = randi([1 W]);
        cy = randi([1 H]);
        meta.spotCenters(k,:) = [cy cx];

        spot = fspecial('gaussian',[H W],opts.spotSigma);
        spot = circshift(spot,[cy-round(H/2) cx-round(W/2)]);
        spot = spot/max(spot(:));

        % Spot height (small local perturbation)
        amp = (rand-0.5)*0.4;   % ±0.2 rad

        pattern = pattern + amp*spot;

    end

    %% -------------------- Store Metadata --------------------

    meta.type = 'step_spot';
    meta.height = opts.height;
    meta.position = opts.position;
    meta.direction = opts.direction;
end
