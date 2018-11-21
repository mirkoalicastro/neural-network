function y = relu(x)
%RELU Relu6
%   x: array di valori o singolo valore
%   y: array di valori o singolo valore
    y = min(max(x, 0), 6);
end

