clear
clc

x = [1,2];
x = dlarray(x);  % means able to track gradient
z= rand(2);
% x = x*z;


y = [1, -3];
dly = dlarray(y);


[fval,gradval, mm] = dlfeval(@get_grad, x, dly, z)  % using a func is a must
y = extractdata(dly)

x = cell(3);
x{1}= dlarray([1,2;2,1]);
x{2} = dly;
x{3} = -dly;
[fval,gradval, mm] = dlfeval(@get_grad2, x, dly, z)  % using a func is a must
y = extractdata(dly)



function [f, grad, m] = get_grad(x, y, z)

xx = x(1);
m = norm(z);
% f = 100*(x(2) - xx.^2).^2 + 10*(1 - xx).^2;
f = 100*(x(2) - x(1).^2).^2 + 10*(1 - x(1)).^2;
f = f + x*y';
grad = dlgradient(f, x);

end

function [f, grad, m] = get_grad2(x, y, z)

m = norm(z);
f = 0;
for i = 1:3
f = f+ x{i}*x{i}';
end
f = f + x{1}*y';
for i = 1:3
grad{i} = dlgradient(f, x{i});
end
end