function [pattern, meta, maskStack] = pattern_shapes(imSize, opts)
%PATTERN_SHAPES Generate a pattern containing multiple random shapes.
%
%   [pattern, meta, maskStack] = pattern_shapes([H,W], opts)
%
%   Optional fields in opts:
%     .numRange        [min max], number of shapes, default [3 6]
%     .types           Cell array of shape types:
%                      {'rotrect','ellipse','circle','triangle'}
%     .sizeRange       [min max], basic shape size (pixels), default [100 150]
%     .ampRange        [amin amax], amplitude range, default [-1 1]
%     .blend           'add'|'max'|'min', blending method, default 'add'
%     .allowClipping   true/false, allow shapes to extend beyond the image,
%                      default false
%     .blurSigma       Gaussian blur sigma (pixels), 0 or empty for no blur
%     .psfKernel       Custom PSF convolution kernel. If provided, it takes
%                      precedence over blurSigma
%     .normalize       'none'|'01'|'phase', output normalization method,
%                      default 'phase'
%     .seed            Random seed for reproducible generation
%
%   Outputs:
%     pattern          Final generated pattern (double)
%     meta             Structure array containing the parameters of each shape
%     maskStack        H x W x N logical array containing the mask of each shape

    arguments
        imSize (1,2) {mustBePositive, mustBeInteger}
        opts.numRange (1,2) double = [3 6]
        opts.types cell = {'rotrect','ellipse','circle','triangle'}
        opts.sizeRange (1,2) double = [100 150]
        opts.ampRange (1,2) double = [-1 1]
        opts.blend char {mustBeMember(opts.blend, ...
            {'add','max','min'})} = 'add'
        opts.allowClipping (1,1) logical = false
        opts.blurSigma (1,1) double = 0
        opts.psfKernel double = []
        opts.normalize char {mustBeMember(opts.normalize, ...
            {'none','01','phase'})} = 'phase'
        opts.seed double = []
    end

    H = imSize(1);
    W = imSize(2);

    % Set the random seed for reproducible generation.
    if ~isempty(opts.seed)
        rng(opts.seed);
    end

    % Randomly determine the number of shapes.
    nShapes = randi([opts.numRange(1), opts.numRange(2)]);

    % Preallocate output variables.
    pattern   = zeros(H, W);
    maskStack = false(H, W, nShapes);

    meta = repmat(struct( ...
        'type', [], 'center', [], 'size', [], 'theta', [], 'amp', []), ...
        nShapes, 1);

    % Generate a coordinate grid for vectorized mask generation.
    [X, Y] = meshgrid(1:W, 1:H);

    for k = 1:nShapes

        % Randomly select the shape type, orientation, amplitude, and size.
        type  = opts.types{randi(numel(opts.types))};
        theta = rand()*180;                           % [0, 180) degrees
        amp   = rand_range(opts.ampRange);

        % Shape size. The interpretation varies slightly among shape types.
        s = rand_range(opts.sizeRange);

        switch type

            case 'rotrect'
                % Rotated rectangle with randomly varied width and height.
                w = s * (0.8 + 0.4*rand());
                h = s * (0.8 + 0.4*rand());

                [cx, cy] = sample_center( ...
                    H, W, w, h, theta, opts.allowClipping);

                M = mask_rotated_rect( ...
                    X, Y, cx, cy, w, h, theta);

                meta(k).size = [h, w];    % Height x width

            case 'ellipse'
                % Rotated ellipse with randomly varied major and minor axes.
                a = s * (0.8 + 0.4*rand());   % Horizontal diameter
                b = s * (0.8 + 0.4*rand());   % Vertical diameter

                [cx, cy] = sample_center( ...
                    H, W, a, b, theta, opts.allowClipping);

                M = mask_rotated_ellipse( ...
                    X, Y, cx, cy, a, b, theta);

                meta(k).size = [b, a];

            case 'circle'
                % Circle with a diameter centered around the specified size.
                dia = s;
                r = dia/2;

                [cx, cy] = sample_center( ...
                    H, W, dia, dia, 0, opts.allowClipping);

                M = ((X-cx).^2 + (Y-cy).^2) <= r^2;

                meta(k).size = [dia, dia];

            case 'triangle'
                % Equilateral triangle with side length approximately s.
                L = s;

                % Estimate the bounding box of the unrotated triangle.
                htri = sqrt(3)/2 * L;

                % Use a conservative bounding-box approximation after rotation.
                approxW = max(L, htri);
                approxH = approxW;

                [cx, cy] = sample_center( ...
                    H, W, approxW, approxH, theta, opts.allowClipping);

                M = mask_equilateral_triangle( ...
                    X, Y, cx, cy, L, theta);

                meta(k).size = [L, L];

            otherwise
                error('Unknown shape type: %s', type);

        end

        % Blend the current shape into the pattern.
        switch opts.blend

            case 'add'
                pattern = pattern + amp * double(M);

            case 'max'
                pattern = max(pattern, amp * double(M));

            case 'min'
                pattern = min(pattern, amp * double(M));

        end

        maskStack(:,:,k) = M;

        meta(k).type   = type;
        meta(k).center = [cy, cx];     % Row, column
        meta(k).theta  = theta;        % Degrees
        meta(k).amp    = amp;

    end

    % Apply optional PSF convolution or Gaussian blur.
    if ~isempty(opts.psfKernel)

        pattern = imfilter( ...
            pattern, opts.psfKernel, 'replicate', 'conv');

    elseif ~isempty(opts.blurSigma) && opts.blurSigma > 0

        pattern = imgaussfilt(pattern, opts.blurSigma);

    end

    % Apply the selected output normalization.
    switch opts.normalize

        case '01'
            pattern = normalize01(pattern);

        case 'phase'
            pattern = 2*pi * normalize01(pattern);

        case 'none'
            % No normalization is applied.

    end

end


%% ========================================================================
% Utility Functions
% =========================================================================

function v = rand_range(rg)
%RAND_RANGE Generate a uniformly distributed random value within a range.

    v = rg(1) + (rg(2)-rg(1))*rand();

end


function A = normalize01(A)
%NORMALIZE01 Normalize an array to the range [0, 1].

    mn = min(A(:));
    mx = max(A(:));

    if mx == mn
        A = zeros(size(A));
    else
        A = (A-mn)/(mx-mn);
    end

end


function [cx, cy] = sample_center(H, W, w, h, thetaDeg, allowClip)
%SAMPLE_CENTER Randomly sample a shape center within the image boundaries.
%
%   When clipping is disabled, the center is selected such that the
%   rotated bounding box remains within the image whenever possible.

    if allowClip

        cx = randi([1, W]);
        cy = randi([1, H]);

        return;

    end

    % Calculate the bounding-box dimensions after rotation.
    th = deg2rad(thetaDeg);

    wbb = abs(w*cos(th)) + abs(h*sin(th));
    hbb = abs(w*sin(th)) + abs(h*cos(th));

    left   = ceil(wbb/2) + 1;
    right  = floor(W - wbb/2);
    top    = ceil(hbb/2) + 1;
    bottom = floor(H - hbb/2);

    if left > right || top > bottom

        % If the shape is too large, allow it to be clipped.
        cx = randi([1, W]);
        cy = randi([1, H]);

    else

        cx = randi([left, right]);
        cy = randi([top, bottom]);

    end

end


function M = mask_rotated_rect(X, Y, cx, cy, w, h, thetaDeg)
%MASK_ROTATED_RECT Generate a rotated rectangular mask.
%
%   w and h denote the width and height of the rectangle.
%   (cx, cy) is the rectangle center.

    th = deg2rad(thetaDeg);

    % Translate the coordinate system to the rectangle center.
    Xc = X - cx;
    Yc = Y - cy;

    % Apply an inverse rotation to obtain an axis-aligned coordinate system.
    Xr = Xc*cos(th) + Yc*sin(th);
    Yr = -Xc*sin(th) + Yc*cos(th);

    M = (abs(Xr) <= w/2) & (abs(Yr) <= h/2);

end


function M = mask_rotated_ellipse(X, Y, cx, cy, a, b, thetaDeg)
%MASK_ROTATED_ELLIPSE Generate a rotated elliptical mask.
%
%   a and b denote the horizontal and vertical diameters, respectively.
%   (cx, cy) is the ellipse center.

    th = deg2rad(thetaDeg);

    Xc = X - cx;
    Yc = Y - cy;

    Xr = Xc*cos(th) + Yc*sin(th);
    Yr = -Xc*sin(th) + Yc*cos(th);

    M = (Xr.^2)/(a/2)^2 + (Yr.^2)/(b/2)^2 <= 1;

end


function M = mask_equilateral_triangle(X, Y, cx, cy, L, thetaDeg)
%MASK_EQUILATERAL_TRIANGLE Generate a rotated equilateral triangle mask.
%
%   L is the side length.
%   (cx, cy) is the triangle center.
%   thetaDeg is the rotation angle in degrees.

    h = sqrt(3)/2 * L;

    % Define the three vertices relative to the geometric center.
    V = [ 0,    2*h/3; ...
         -L/2, -h/3; ...
          L/2, -h/3];

    % Rotate the vertices.
    th = deg2rad(thetaDeg);

    R = [cos(th), -sin(th); ...
         sin(th),  cos(th)];

    V = (R * V')';

    % Translate the vertices to the specified center.
    V(:,1) = V(:,1) + cx;
    V(:,2) = V(:,2) + cy;

    % Construct the mask using the polygon vertices.
    xv = V(:,1);
    yv = V(:,2);

    M = inpolygon(X, Y, xv, yv);

end

