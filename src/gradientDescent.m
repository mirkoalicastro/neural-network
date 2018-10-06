function [net, deltaW, deltaB] = gradientDescent(net, dW, dB, eta)
%GRADIENTDESCENT
%   *net: rete neurale feed forward fully connected
%   *dW: cell array contenente le derivate della funzione di errore
%        dei pesi divise per layer della rete
%   *dB: cell array contenente le derivate della funzione di errore
%        dei bias dei nodi divise per layer della rete
%   *eta: parametro attraverso il quale controllo l'aggiornamento
%         dei pesi/bias in congiunzione con la valutazione delle 
%         derivate della funzione di errore rispetto ai pesi
%   *net: la net in input con pesi/bias aggiornati
%   *deltaW: scostamenti dei pesi effettuati
%   *deltaB: scostamenti dei bias effettuati

%Riferimenti: lezioni frontali, Bishop

%La funzione permette di aggironare i pesi mediante la tecnica
%della discesa del gradiente dopo aver calcolate le 
%derivate della funzione di errore rispetto ai pesi


%Inizializzazione dei cell array forniti in output dalla
%funzione per motivi di performance
    deltaW = cell(1, net.numLayers);
    deltaB = cell(1, net.numLayers);

    for layer=1:net.numLayers
        deltaW{layer} = eta * dW{layer};
        deltaB{layer} = eta * dB{layer};
        net.B{layer} = net.B{layer} - deltaB{layer};
        net.W{layer} = net.W{layer} - deltaW{layer};
    end
end