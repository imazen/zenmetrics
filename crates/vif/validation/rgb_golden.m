pkg load image;
% RGB golden — vifp_mscale is single-channel, so the house `vif_rgb8`
% convention (unrounded 0.2989/0.5870/0.1140 luma) is verified by
% computing the luma plane explicitly and running the reference on it.
r = zeros(64,64,3); d = zeros(64,64,3);
r(:,:,1)=gen(1,64,64); r(:,:,2)=gen(2,64,64); r(:,:,3)=gen(3,64,64);
d(:,:,1)=gen(4,64,64); d(:,:,2)=gen(5,64,64); d(:,:,3)=gen(1,64,64);
lr = 0.2989*double(r(:,:,1)) + 0.5870*double(r(:,:,2)) + 0.1140*double(r(:,:,3));
ld = 0.2989*double(d(:,:,1)) + 0.5870*double(d(:,:,2)) + 0.1140*double(d(:,:,3));
printf("rgb_g123_g451_64 64 64 vifp=%.12f\n", vifp_mscale(lr, ld));
