
clear all, close all, clc

%% Load data

gfivalue = 2.4;

rawDATA = importdata(['./initial_conditions/Raw/data_g' num2str(gfivalue*100) '.dat']);

n = 512;
m = 512;

U = reshape(rawDATA(:,1),n,m);
V = reshape(rawDATA(:,2),n,m);
W = reshape(rawDATA(:,3),n,m);

newn = 500;
newm = 500;

dn = floor((n-newn)/2);
dm = floor((m-newm)/2);

U = U((dn+1):(n-dn),(dm+1):(m-dm));
V = V((dn+1):(n-dn),(dm+1):(m-dm));
W = W((dn+1):(n-dn),(dm+1):(m-dm));

figure;
imagesc(U)
title('Counterclock')
xlabel('X')
ylabel('Y')
hold on

DU3d1 = reshape(U,newm*newn,1);
DV3d1 = reshape(V,newm*newn,1);
DW3d1 = reshape(W,newm*newn,1);

% DW3d1 = reshape(DW3d,newm*newn*zpixels,1);

%% Rewrite in my software format

Dout = cat(2,DU3d1,DV3d1,DW3d1);

%% Save data

text = ['./initial_conditions/readyCUDA/init_g' num2str(gfivalue) '.dat'];

save(text, 'Dout', '-ascii')


