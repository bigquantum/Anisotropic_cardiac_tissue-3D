clear all, close all, clc

addpath('../image_improvement');

%% Load data
dataTipraw = importdata('./DATA/EPIENDO1/dataTip.dat');
dataTipGradraw = importdata('./DATA/EPIENDO1/dataGrad.dat');
dataTipSizeraw = importdata('./DATA/EPIENDO1/dataTipSize.dat');
 
fid = fopen('./DATA/EPIENDO1/dataparam.csv');
parameters = textscan(fid,'%s%s','delimiter',',');

parameters{1,:}
tablehight = size(parameters{1,1},1);
table = zeros(tablehight,1);

for i = 2:tablehight(1,1)
    table(i) = str2num(cell2mat(parameters{:,2}(i)));
end

%% Rescale to physical coordinates

p.dx = table(6);
p.dy = table(7);
p.dz = table(8);
p.Lx = table(9);
p.Ly = table(10);
p.Lz = table(11);

xtip = (dataTipraw(:,1) - 1)*p.dx - p.Lx/2;
ytip = (dataTipraw(:,2) - 1)*p.dy - p.Ly/2;
ztip = (dataTipraw(:,3) - 1)*p.dz - p.Lz/2;

gx = (dataTipGradraw(:,1) - 1)*p.dx - p.Lx/2;
gy = (dataTipGradraw(:,2) - 1)*p.dy - p.Ly/2;
gz = (dataTipGradraw(:,3) - 1)*p.dz - p.Lz/2;
gradx = dataTipGradraw(:,4);
grady = dataTipGradraw(:,5);
gradz = dataTipGradraw(:,6);

%% Run animation

figure;
for i = 1:(size(dataTipSizeraw,1)-1)
   sizel = dataTipSizeraw(i) + 1;
   sizeu = dataTipSizeraw(i+1);
   interval = sizel:sizeu;
   x = xtip(interval,:);
   y = ytip(interval,:);
   z = ztip(interval,:);
   scatter3(x,y,z,'MarkerEdgeColor','k',...
        'MarkerFaceColor',[0 .75 .75])
   title('Filament 3D')
   grid on
   axis([-p.Lx/2 p.Lx/2 -p.Ly/2 p.Ly/2 -p.Lz/2 p.Lz/2])
   pause(0.1)
end

%% Sorted elements 

figure;
for i = 1:(size(dataTipSizeraw,1)-1)
    newxyz = [];
    sizel = dataTipSizeraw(i) + 1;
    sizeu = dataTipSizeraw(i+1);
    interval = sizel:sizeu;
    xyz = [xtip(interval,:) ytip(interval,:) ztip(interval,:)];
    for j = 1:size(interval,2)
        point = xyz(j,:);
        if j == 1
            newxyz = point;
        end
        pointvec = repmat(point,size(interval,2),1);
        distxyz = sqrt(sum((xyz - pointvec) .^ 2,2));
        distxyz(distxyz == 0 ) = NaN;
        [~,idx] = min(distxyz);
        newxyz = [newxyz ; xyz(idx,:)];
    end
%     disp('raw')
%     size(interval)
%     disp('sort')
%     size(newxyz)
%     n = 100;
% t = linspace(0,1,n)';
% P = (1-t)*P1 + t*P2;
    x = xyz(:,1);
    y = xyz(:,2);
    z = xyz(:,3);
    scatter3(x,y,z,'MarkerEdgeColor','k',...
        'MarkerFaceColor',[0 .75 .75])
    title('Filament 3D')
        view(90,90)
    grid on
    axis([-p.Lx/2 p.Lx/2 -p.Ly/2 p.Ly/2 -p.Lz/2 p.Lz/2])
    pause(0.1)
end

%% Epi-endo-myo

txyzmin = [];
txyzmax = [];
halfpoint = 0.0;

figure;
for i = 1:(size(dataTipSizeraw,1)-1)
    newxyz = [];
    sizel = dataTipSizeraw(i) + 1;
    sizeu = dataTipSizeraw(i+1);
    interval = sizel:sizeu;
    xyz = [xtip(interval,:) ytip(interval,:) ztip(interval,:)];
    % Epicardium & endocardium
    minidx = find(xyz(:,3) == min(xyz(:,3)));
    maxidx = find(xyz(:,3) == max(xyz(:,3)));
    x = xyz(:,1);
    y = xyz(:,2);
    z = xyz(:,3);
    txyzmin = [txyzmin ; x(minidx) y(minidx)];
    txyzmax = [txyzmax ; x(maxidx) y(maxidx)];
    scatter(txyzmin(:,1),txyzmin(:,2),'MarkerEdgeColor','k','MarkerFaceColor','b')
    hold on
    scatter(txyzmax(:,1),txyzmax(:,2),'MarkerEdgeColor','k','MarkerFaceColor','r')
    % Myocardium;
    idxdown = find((z < halfpoint) ,1,'last');
    idxup = find(z > halfpoint ,1,'first');
    P1 = [x(idxdown) y(idxdown) z(idxdown)];
    P2 = [x(idxup) y(idxup) z(idxup)];
    t = (halfpoint - z(idxdown)) / (z(idxup) - z(idxdown));
    P0 = (1-t).*P1 + t.*P2;
    hold on
    scatter(P0(:,1),P0(:,2),'MarkerEdgeColor','k','MarkerFaceColor','m')
    title('Endocardium & Epicardium')
    grid on
%     axis([-p.Lx/2 p.Lx/2 -p.Ly/2 p.Ly/2])
    pause(0.1)
end

