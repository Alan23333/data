
clear all;clc;close all;

F = [2  1  1  1  1  1  0  0  0  0  0  0  0  0  0  0  2  6 11 17 21 22 21 20 20 19 19 18 18 17 17;...
     1  1  1  1  1  1  2  4  6  8 11 16 19 21 20 18 16 14 11  7  5  3  2  2  1  1  2  2  2  2  2;...
     7 10 15 19 25 29 30 29 27 22 16  9  2  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1];

for band = 1:3
    div = sum(F(band, :));
    for i = 1:31
        F(band, i) = F(band, i) / div;
    end
end

sf = 8;

sz1 = [96 96];
sz2 = [512 512];
s0 = 1;
psf = fspecial('gaussian', 8, 2);
B1 = psf2otf(psf, sz1);
B2 = psf2otf(psf, sz2);

F = F(:, 3:31);
for band = 1:size(F, 1)
    div = sum(F(band, :));
    for i = 1:size(F, 2)
        F(band, i) = F(band, i) / div;
    end
end
R = F';

