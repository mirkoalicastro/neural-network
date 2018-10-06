function y = calculateErrorOn(net, data, labels, errorFunction, afterProcessFunction)
%CALCULATEERRORON 
%   *net: rete neurale feed forward fully connected
%   *data: input da propagare nella rete
%   *labels: etichette degli elementi dati in input
%   *errorFunction: funzione di errore da utilizzare per valutare
%                   i risultati della classificazione
%   *afterProcessFunction: funzione di post processing dell'output della rete


%Funzione che sfrutta la specifica funzione di errore errorFunction
%per calcolare l'errore di classificazione commesso dalla rete


    [~, z] = forwardProp(net, data, afterProcessFunction);

%Valuto l'errore di classificazione come somma degli errori commessi 
%dai singoli nodi di output
    y = sum(errorFunction(z{net.numLayers}, labels));
end

