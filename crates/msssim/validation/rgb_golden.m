pkg load image;
K = [0.01 0.03]; win = fspecial('gaussian', 11, 1.5);
W5 = [0.0448 0.2856 0.3001 0.2363 0.1333];
r = zeros(64,64,3); d = zeros(64,64,3);
r(:,:,1)=gen(1,64,64); r(:,:,2)=gen(2,64,64); r(:,:,3)=gen(3,64,64);
d(:,:,1)=gen(4,64,64); d(:,:,2)=gen(5,64,64); d(:,:,3)=gen(1,64,64);
lr = 0.2989*double(r(:,:,1)) + 0.5870*double(r(:,:,2)) + 0.1140*double(r(:,:,3));
ld = 0.2989*double(d(:,:,1)) + 0.5870*double(d(:,:,2)) + 0.1140*double(d(:,:,3));
printf("rgb_g123_g451_64 64 64 L=3 msssim=%.12f\n", msssim(lr, ld, K, win, 3, W5(1:3), 'product'));
