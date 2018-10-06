function y = softmax(x)
%SOFTMAX
%   x: array o singolo valore 
%   y: array contenente i valori processati dal softmax
%       o valore singolo


%Calcola la funzione di softmax su un insieme di valori


    y = exp(x) ./ sum(exp(x),2);
end

