function phi = unwrap_WLS_FFT(phi_wrapped, weight)

% phi_wrapped : wrapped phase (-pi,pi]
% weight      : weight map (same size as phase), optional

if nargin < 2
    weight = ones(size(phi_wrapped));
end

[M,N] = size(phi_wrapped);

% wrap operator
wrap = @(x) atan2(sin(x),cos(x));

% phase differences
dx = wrap(diff(phi_wrapped,1,2));
dy = wrap(diff(phi_wrapped,1,1));

% pad gradients
dx = [dx zeros(M,1)];
dy = [dy; zeros(1,N)];

% weighted gradients
wx = weight .* dx;
wy = weight .* dy;

% divergence
div = zeros(M,N);
div(:,1:end-1) = div(:,1:end-1) + wx(:,1:end-1);
div(:,2:end)   = div(:,2:end)   - wx(:,1:end-1);

div(1:end-1,:) = div(1:end-1,:) + wy(1:end-1,:);
div(2:end,:)   = div(2:end,:)   - wy(1:end-1,:);

% solve Poisson equation using FFT
[X,Y] = meshgrid(0:N-1,0:M-1);
denom = (2*cos(2*pi*X/N)-2) + (2*cos(2*pi*Y/M)-2);
denom(1,1) = 1;

phi = real(ifft2( fft2(div) ./ denom ));
phi = phi - phi(1,1); % remove piston
end