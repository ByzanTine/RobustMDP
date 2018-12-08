clear;
load values_classic v_nominal v_robust;

storm_state = 1;

vn = squeeze(v_nominal(storm_state, :, :));
vr = squeeze(v_robust(storm_state, :, :));
imagesc([vn ones(11, 1) * max(vr(:)) vr]);
set(gca, 'xtick', []);
set(gca, 'ytick', []);
for i = 1:11
    for j = 1:11
        text(j, i, sprintf('%d', round(vn(i, j))));
        text(j + 12, i, sprintf('%d', round(vr(i, j))));
    end;
end;
