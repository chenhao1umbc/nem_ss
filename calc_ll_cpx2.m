function ll = calc_ll_real(x,vhat,Rj,Rb)

[M,N,F] = size(x);
J = size(vhat,3);

ll = 0;

for n = 1:N
    for f = 1:F
        Rxnf = Rb;
        for j1 = 1:J
            Rcjnf = vhat(n,f,j1)*Rj(:,:,j1);
            Rxnf = Rxnf + Rcjnf;
        end
        %ll = ll - 0.5*log(2*pi*det(Rxnf)) - x(:,n,f)'*inv(Rxnf)*x(:,n,f)/2; % real
        ll = ll - log(pi*det(Rxnf)) - x(:,n,f)'*inv(Rxnf)*x(:,n,f);  % complex
        assert(abs(imag(ll)) < 1e-5);
        ll = real(ll);
    end
end