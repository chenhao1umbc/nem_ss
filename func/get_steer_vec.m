function steer_vec = get_steer_vec(aoa, n_channel, J)
eps = 1e-30;
elementPos = (0:.1:n_channel*0.1-0.1);
c = physconst('LightSpeed');
fc = 1e9;
lam = c/fc;

steer_vec = zeros(J ,n_channel); %[n_sources, n_channel)
for i = 1:J
    ang = [aoa(i);0];
    sv = steervec(elementPos/lam,ang);
    vec = reshape(real(sv),[1, n_channel]) + eps;
    steer_vec(i, :) = vec/ norm(vec);
end

end