% Per-channel + BT.601-luma goldens for the rgb8/luma8 Rust paths.
% The reference is single-channel; per-channel goldens + their mean
% verify the multi-plane convention end-to-end.
% NOTE: gen() returns uint64 — cast to double BEFORE the luma mix or
% Octave does integer arithmetic (0.299*uint64(224) == 67).
g  = gen(2,32,24);          % same construction as the Rust test
rR = g; rG = mod(g+40,256); rB = floor(g/2);
dR = mod(g+9,256); dG = mod(g+31,256); dB = mod(rB+7,256);
rung('rgb_R', rR, dR);
rung('rgb_G', rG, dG);
rung('rgb_B', rB, dB);
lum_r = round(0.299*double(rR) + 0.587*double(rG) + 0.114*double(rB));
lum_d = round(0.299*double(dR) + 0.587*double(dG) + 0.114*double(dB));
rung('rgb_luma', lum_r, lum_d);
