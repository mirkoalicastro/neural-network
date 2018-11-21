function y = relu6Dx(x)
%RELU6DX Relu derivative
    y = x;
    y(x>0 & x<6) = 1;
    y(x<=0 | x>=6) = 0;
end

