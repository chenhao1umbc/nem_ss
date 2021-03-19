function steer_vec = get_steer_vec(aoa, n_channel)
eps = 1e-30;
elementPos = (0:.1:.5);
c = physconst('LightSpeed');
fc = 1e9;
lam = c/fc;
ang = [aoa;0];
sv = steervec(elementPos/lam,ang);
steer_vec = reshape(real(sv),[1, n_channel]) + eps;
end