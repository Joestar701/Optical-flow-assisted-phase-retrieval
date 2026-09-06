function phaseMap = generate_phase_map(imSize)
%GENERATE_PHASE_MAP Generate a random cylindrical phase map with Airy PSF blur.
%
%   imSize    : Output matrix size, e.g., [512, 512].
%   phaseMap  : Generated phase distribution in the range [0, 2*pi].

    %% Step 1: Initialize the plane
    plane = zeros(imSize);

    %% Step 2: Generate random cylinders
    numCylinders = randi([20, 30]);   % Random number of cylinders

    [X, Y] = meshgrid(1:imSize(2), 1:imSize(1));

    for i = 1:numCylinders

        cx = randi([1, imSize(2)]);   % Random cylinder center
        cy = randi([1, imSize(1)]);
        radius = randi([1, 3]);       % Random radius
        height = 0.5 + 0.5*rand();    % Random height in [0.5, 1.0]

        % Define the cylindrical region
        mask = (sqrt((X-cx).^2 + (Y-cy).^2) <= radius);

        plane(mask) = height;

    end

    %% Step 3: Apply Airy PSF blur
    psfKernel = airy_PSF(256);        % Generate a 256 x 256 Airy PSF

    blurred = imfilter( ...
        plane, psfKernel, 'replicate', 'conv');

    %% Step 4: Normalize to the phase range [0, 2*pi]
    phaseMap = 2*pi*normalize_to_01(blurred);

end


%% ========================================================================
% Generate Airy PSF
% =========================================================================
function psf = airy_PSF(m)
%AIRY_PSF Generate a diffraction PSF for a circular aperture.
%
%   m   : Size of the pupil grid, typically 256.
%   psf : Normalized Airy PSF.

    raDiusOut = randi([16, 64]);   % Random outer radius
    raDiusIn = 0;                  % Inner radius

    mapSpace = ones(m);            % Pupil function
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

    % Compute MTF and PSF
    mtf = conv2(mapSpace, mapSpace);

    mtf_fft = fftshift( ...
        fft2(fftshift(mtf)));

    psf = abs(mtf_fft).^2;

    % Normalize the PSF
    psf = psf / max(psf(:));

end


%% ========================================================================
% Normalize a Matrix to [0, 1]
% =========================================================================
function normalized = normalize_to_01(matrix)
%NORMALIZE_TO_01 Normalize the input matrix to the range [0, 1].

    minVal = min(matrix(:));
    maxVal = max(matrix(:));

    if maxVal == minVal
        normalized = zeros(size(matrix));
    else
        normalized = (matrix - minVal) / (maxVal - minVal);
    end

end

