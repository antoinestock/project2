clc
clear all
close all
%loading data
[p,x1,thread,x2,n,x3,time,x4] = textread('job1.txt','%s%d%s%d%s%d%s%f','delimiter','=');

figure(1)
plot(x3(1:14), x4(1:14), '-*r','linewidth', 2);

hold on;
plot(x3(15:28), x4(15:28), '-*g','linewidth', 2);
plot(x3(29:42), x4(29:42), '-*k','linewidth', 2);
plot(x3(43:56), x4(43:56), '-*b','linewidth', 2);
plot(x3(57:70), x4(57:70), '--xr','linewidth', 2);
plot(x3(71:84), x4(71:84), '--xg','linewidth', 2);
plot(x3(85:98), x4(85:98), '--xc','linewidth', 2);
plot(x3(99:112), x4(99:112), '--xb','linewidth', 2);
legend('Process=1', 'Process=2','Process=6','Process=12','Process=18','Process=24','Process=30','Process=36');
xlabel('Size of the system');
ylabel('Computation time [sec]');
hold off;
grid on

figure(2)
N=x3(1:14);
plot(N, N.^2.*log(N), 'b','linewidth', 2);
xlabel('Size of the system');
ylabel('Complexity');
grid on

T=zeros(8,14);
for i=1:8
    for j=1:14
        T(i,j)=x4(14*(i-1)+j);
    end
end

P=[1,2,6,12,18,24,30,36];

figure(3)
for i=1:14
    plot(P,T(1,i)./T(:,i),'linewidth', 2);
    hold on
end
xlabel('Number of processes');
ylabel('Speedup');
legend('N=2', 'N=4','N=8','N=16','N=32','N=64','N=128','N=256','N=512','N=1024','N=2048','N=4096','N=8192','N=16384');
grid on
hold off


figure(4)
for i=1:14
    plot(P,T(1,i)./T(:,i)./P(:),'linewidth', 2);
    hold on
end
xlabel('Number of processes');
ylabel('Efficiency');
legend('N=2', 'N=4','N=8','N=16','N=32','N=64','N=128','N=256','N=512','N=1024','N=2048','N=4096','N=8192','N=16384');
grid on
hold off;

