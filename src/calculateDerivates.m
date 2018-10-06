function [dW, dB] = calculateDerivates(net, delta, TrS, z)
%CALCULATEDERIVATES
%   *net: rete neurale feed forward fully connected
%   *delta: cell array contenenete i delta k dei nodi suddivisi per layer
%   *TrS: training set con il quale addestrare la rete. E'una matrice
%         in cui gli elementi sono disposti su righe e dunque la colonna
%         j-esima rappresenta l'espressione della caratteristica j-esima
%         dello specifico elemento 
%   *z: cell array contenente i valori di output dei nodi suddivisi per layer
%   *dW: cell array contenente le derivate della funzione di errore
%        dei pesi divise per layer della rete
%   *dB: cell array contenente le derivate della funzione di errore
%        dei bias dei nodi divise per layer della rete

%Riferimenti: lezioni frontali, Bishop

%Permette di calcolare le derivate dell'errore rispeto al peso wj
%grazie ai delta calcolati grazi alla back propagation


%Per il primo strato gli zi del layer precedente corrispondono alle
%caratteristiche in input
    prev = TrS;

%Inizializzazione delle strutture (per motivi di performance)
    dW = cell(1, net.numLayers);
    dB = cell(1, net.numLayers);


    for layer=1:net.numLayers
        
%Per l'aggiornamento dei bias gli zi valgono 1. E' possibile 
%dunque procedere direttamente con la somma dei deltaj 
        dB{layer} = sum(delta{layer},1);

        dW{layer} = delta{layer}' * prev;
        prev = z{layer};
    end
end