function net = myNet(numFeatures, numNodes, functions, functionsDx)
%MYNET
%  numFeatures: numero di feature in input alla rete
%  numNodes: array che descrive il numero di nodi per ogni
%            layer della rete
%  functions: array che descrive le funzioni di output per ogni
%             layer della rete
%  functionsDx: array che descrive le derivate delle funzioni
%               di output per ogni layer della rete
%  *net: rete neurale feed forward fully connected con funzione di output
%        per ogni livello dettate da functions e di attivazione lineare

%Riferimenti: lezioni frontali, Bishop


%La seguente funzione permette di creare la struttura basilare di una rete 
%neurale feed forward fully connected con funzioni di output stabilite
%dall'utente

    prev = numFeatures;

%Memorizzo il numero totale di layers di cui la rete sara' composta
    net.numLayers = size(numNodes, 2);

%Range entro il quale ricadranno i pesi iniziali
    max = 0.07;
    min = -0.09;


%Memorizzo il numero totale di features che la rete processera'
    net.numFeatures = numFeatures;

%Creazione dei pesi delle connessioni in uscita dai nodi e creazione
%dei bias come nodi il cui output e' sempre 1 e pesi delle connessioni 
%dati dal valore stesso dei bias
%Vengono inoltre memeorizate le funzioni e le derivate per ogni layer
    for i=1 : net.numLayers
        net.W{i} = (max-min) .* rand(numNodes(i), prev) + min;
        net.B{i} = (max-min) .* rand(1,numNodes(i)) + min;
        
        net.F{i} = functions{i};
        net.Fdx{i} = functionsDx{i};
        prev = numNodes(i);
    end
    net.numOutput = prev;
end

