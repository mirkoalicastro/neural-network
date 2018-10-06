function [y, GL] = generalizationLoss(errTS, errVS, epoch, minErrTS, minErrVS, threshold)
%GENERALIZATIONLOSS
%   errTS: array contenente le informazioni sull'errore commesso sul
%          training set ad ogni epoca durante l'addestramento
%   errTS: array contenente le informazioni sull'errore commesso sul
%          validation set ad ogni epoca durante l'addestramento
%   epoch:  epoca attuale durante l'addestramento
%   minErrTS: errore minimo compiuto sul training set durante il  training
%   minErrVS: errore minimo compiuto sul validation set durante il  training
%   threshold: soglia di perdita di generalizzazione sopra la quale la fase
%   di treaning deve terminare
%   *y: booleano che indica all'algoritmo di apprendimento se fermarsi o meno
%   *GL: perdita di generalizzazione all'epoca attuale

% Riferimenti: lezioni frontali, paper Early Stopping | but when? pg.4


%Implementazione della funzione di generalization loss utile a capire quando
%terminare l'apprendimento poiche' si sta facendo overfitting sul training set


    y = false;
    if ~exist('threshold','var')
        threshold = 0.65;
    end
%GL di epoca t e' definita come l'incremento relativo dell'errore
%sul validation set rispetto al minimo registrato
    GL = (errVS(epoch)/minErrVS - 1);

    if(GL > threshold)
        y = true;
    end
end

