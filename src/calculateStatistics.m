function confusionMatrix = calculateStatistics(net, TeS, TeL, onlyOneFunction)
%CALCULATESTATISTICS
%   *net: rete neurale feed forward fully connected
%   *TeS: test set per la rete. E'una matrice
%         in cui gli elementi sono disposti su righe e dunque la colonna
%         j-esima rappresenta l'espressione della caratteristica j-esima
%         dello specifico elemento 
%   *TeL: insieme delle label degli elementi del test set. Alla riga
%         i-esima del TeS corrisponde la Label i-esima
%   *onlyOneFunction: funzione di post processing dell'output della rete
%   *confusionMatrix: matrice di confusione per valutare le performance
%                     di classificazione


%Restituisce la matrice di confusione rispetto i risultati ottenuti
%dalla rete sul test set al fine di valutarne le prestazioni


%Attivazione della rete col test set ed estrazione dell'output
    [~, z] = forwardProp(net, TeS, onlyOneFunction);
    output = z{net.numLayers};

%Preallocazione matrice per motivi di performance
    confusionMatrix = zeros(size(TeL,2),size(TeL,2));

%Calcolo della matrice di confusione analizzando l'output della rete
%e la label corrispondente a seguito della propagazione di tutti gli
%elementi del test set
    for i=1:size(TeL,2)
        tmp_label = TeL(:, i);
        for j=1:size(TeL,2)
            tmp_output = output(:, j);
            confusionMatrix(i,j) = sum((tmp_label+tmp_output)>1);
        end
    end
end