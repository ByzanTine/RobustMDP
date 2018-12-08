clear;
load GS_11_K_1_RUN_0_G2G_0.90_B2B_0.10_epsilon_sweep.mat ...
    epsilonrecord ret_entropy ret_likelihood ret_nominal;
%%
ret_likelihood = double(ret_likelihood);
ret_nominal = double(ret_nominal);

mean_likelihood = mean(ret_likelihood, 2);
mean_nominal = mean(ret_nominal, 2);

plot(epsilonrecord, mean_nominal, 'r-');
hold on;
plot(epsilonrecord, mean_likelihood, 'b-');

grid on;

%%
clear;
load COST_5.00_GS_11_K_1_RUN_0_G2G_0.90_B2B_0.10_tol2_sweep.mat ...
    epsilonrecord ret_entropy ret_likelihood ret_nominal;
%% 
ret_likelihood = double(ret_likelihood);
ret_nominal = double(ret_nominal);

mean_likelihood = mean(ret_likelihood, 2);
mean_nominal = mean(ret_nominal, 2);

plot(epsilonrecord, mean_nominal, 'r-');
hold on;
plot(epsilonrecord, mean_likelihood, 'b-');

set(gca, 'XScale', 'log');

grid on;

