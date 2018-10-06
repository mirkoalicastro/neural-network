function [a,z] = forwardProp(net, TrS, afterProcessFunction)
%FORWARDPROP
%   *net: rete neurale feed forward fully connected
%   *TrS: training set con il quale addestrare la rete. E'una matrice
%         in cui gli elementi sono disposti su righe e dunque la colonna
%         j-esima rappresenta l'espressione della caratteristica j-esima
%         dello specifico elemento 

%Riferimenti: lezioni frontali, Bishop

%Permette la propagazione dell'input dal primo strato fino allo strato di
%output di una rete neurale feed forward fully connected costrtuita con myNet


%Inizializzazione delle strutture (per motivi di performance)
%contenenti rispettivamente i valori di attivazione e
%di uscita dei nodi divisi per layer
    a = cell(1, net.numLayers);
    z = cell(1, net.numLayers);

%Per il primo strato l'output del layer precedente corrisponde alle
%caratteristiche in input
    prev = TrS;
    for layer=1:net.numLayers
        a{layer} = (prev * net.W{layer}') + net.B{layer};
        z{layer} = net.F{layer}(a{layer});
        prev = z{layer};
    end

%Possibilitá di effettuare un post processing delle rete
%come ad esempio SOFTMAX
    z{net.numLayers} = afterProcessFunction(z{net.numLayers});
end