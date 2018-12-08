clear;
load GS_9_K_4_RUN_1_G2G_0.90_B2B_0.10_sweep_200by1.mat;

ret_nominal = squeeze(ret_nominal);
ret_robust = squeeze(ret_robust);

en = mean(ret_nominal, 2);
er = mean(ret_robust, 2);

sn = std(ret_nominal')';
sr = std(ret_robust')';

errorbar(epsilon_record, en, sn, 'LineWidth', 3, 'CapSize', 0, 'color', 'y');
hold on;
errorbar(epsilon_record, er, sr, 'LineWidth', 3, 'CapSize', 0, 'color', 'c');
pn = plot(epsilon_record, en, 'r-', 'LineWidth', 1.5);
pr = plot(epsilon_record, er, 'b-', 'LineWidth', 1.5);
xlim([-0.01 0.9]);
legend([pn, pr], 'Nominal', 'Robust');
xlabel('\epsilon');
ylabel('cost');
grid on;

%%
clear;
load GS_9_K_4_RUN_1_G2G_0.90_B2B_0.10_sweep_1by200.mat;

ret_nominal = squeeze(ret_nominal);
ret_robust = squeeze(ret_robust);

en = mean(ret_nominal, 2);
er = mean(ret_robust, 2);

sn = std(ret_nominal')';
sr = std(ret_robust')';

errorbar(cost_record, en, sn, 'LineWidth', 3, 'CapSize', 0, 'color', 'y');
hold on;
errorbar(cost_record, er, sr, 'LineWidth', 3, 'CapSize', 0, 'color', 'c');
pn = plot(cost_record, en, 'r-', 'LineWidth', 1.5);
pr = plot(cost_record, er, 'b-', 'LineWidth', 1.5);
% xlim([-0.01 0.9]);
legend([pn, pr], 'Nominal', 'Robust');
xlabel('c');
ylabel('cost');
grid on;