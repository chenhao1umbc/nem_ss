clear
clc

x = [1,2];
x = dlarray(x);  % means able to track gradient
y = [1, -3];
dly = dlarray(y);
[fval,gradval] = dlfeval(@get_grad,x, dly)  % using a func is a must
y = extractdata(dly)

function [f,grad] = get_grad(x, y)

f = 100*(x(2) - x(1).^2).^2 + (1 - x(1)).^2;
f = f + x*y';
grad = dlgradient(f,y);

end