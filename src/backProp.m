function [dW, dB] = backProp(net, TrS, TrL, errorFunctionDx, afterProcessFunction)
%BACKPROP
%   *net: rete neurale feed forward fully connected
%   *TrS: training set con il quale addestrare la rete. E'una matrice
%         in cui gli elementi sono disposti su righe e dunque la colonna
%         j-esima rappresenta l'espressione della caratteristica j-esima
%         dello specifico elemento 
%   *TrL: insieme delle label degli elementi del training set. Alla riga
%         i-esima del TrS corrisponde la Label i-esima
%   *errorFunctionDx: derivata della funzione di errore rispetto un peso
%   *afterProcessFunction: funzione di post processing dell'output della rete
%   *dW: cell array contenente le derivate della funzione di errore
%        dei pesi divise per layer della rete
%   *dB: cell array contenente le derivate della funzione di errore
%        dei bias dei nodi divise per layer della rete

%Riferimenti: lezioni frontali, Bishop

%Funzione che si occupa di calcolare le derivate della funzione di errore
%rispetto i singoli i nodi mediante la tecnica della back propagation


%Propagazione dell'input della rete e storage dei valori di attivazione e
%e di uscita dei singoli nodi divisi per layer
    [a, z] = forwardProp(net, TrS, afterProcessFunction);

%Inizializzazione della struttura (per motivi di performance)
%contenete i delta utili al calcolo delle derivate
    delta = cell(1, net.numLayers);

%Calcolo dei delta per i nodi dello strato di output
    delta{net.numLayers} = net.Fdx{net.numLayers}(a{net.numLayers}) .* errorFunctionDx(z{net.numLayers}, TrL);

%Calcolo dei delta per tutti i nodi dei layer hidden (e per i loro bias)
    for layer=net.numLayers-1:-1:1
       delta{layer} = net.Fdx{layer}(a{layer}) .* (delta{layer+1} * net.W{layer+1});
    end

%Concludo la back prop sfruttando i delta e i valori di uscita del layer precedente
    [dW, dB] = calculateDerivates(net, delta, TrS, z);
end