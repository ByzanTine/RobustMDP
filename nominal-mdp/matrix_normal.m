clear;
load GS_9_K_4_RUN_1_G2G_0.90_B2B_0.10_sweep.mat;
rn = mean(ret_nominal, 3);
rn = rn(1:14, 1:14);
rr = mean(ret_robust, 3);
rr = rr(1:14, 1:14);

imagesc([rn ones(14, 1) * max(rn(:)) rr]);
set(gca, 'xtick', []);
set(gca, 'ytick', []);
for i = 1:14
    for j = 1:14
        text(j, i, sprintf('%d', round(rn(i, j))));
        text(j + 15, i, sprintf('%d', round(rr(i, j))));
    end;
end;
for i = 1:14
    text(-1, i, sprintf('%.3f', epsilon_record(i)));
    text(30, i, sprintf('%.3f', epsilon_record(i)));
    text(i, 0, sprintf('%d', round(cost_record(i))));
    text(i + 15, 0, sprintf('%d', round(cost_record(i))));
end;