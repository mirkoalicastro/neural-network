function y = crossEntropyDx(X,Y)
%CROSSENTROPYDX 
%   *X: singolo valore o array
%   *Y: singolo valore o array 

%Riferimenti: lezioni frontali, Bishop

%Derivata della Cross Entropy nel caso in cui
%si utilizzi SoftMax per il post-processing della rete


    y = X - Y;

end

