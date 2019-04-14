clc
clear all
close all
%loading data
k=2:14;
error = [1.927711e-1,4.476185e-2,1.098931e-2,2.734955e-3,6.829684e-4,1.706940e-4,4.267049e-5,1.066744e-5,2.666847e-6,6.667092e-7,1.666619e-7,4.180118e-8,1.130315e-8];

figure(1)
loglog(2.^(-2*k),error, '-*b','linewidth', 2);

xlabel('h^2');
ylabel('Pointwise error');
grid on




