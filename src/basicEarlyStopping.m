function y = basicEarlyStopping(errTS, errVS, epoch, minErrTS, minErrVS)
%BASICEARLYSTOPPING
%   errTS: array contenente le informazioni sull'errore commesso sul
%          training set ad ogni epoca durante l'addestramento
%   errTS: array contenente le informazioni sull'errore commesso sul
%          validation set ad ogni epoca durante l'addestramento
%   epoch:  epoca attuale durante l'addestramento
%   minErrTS: errore minimo compiuto sul training set durante il  training
%   minErrVS: errore minimo compiuto sul validation set durante il  training
%   *y: booleano che indica all'algoritmo di apprendimento se fermarsi o meno

% Riferimenti: lezioni frontali, paper Early Stopping | but when? criterio pg.2


%Implemententazione del piu' semplice dei criteri di early stopping 


    y = false;
    
%Appena l'errore sul validation set durante l'attuale epoca diventa maggiore
% rispetto al precedente ferma il training
    if(epoch > 1 && errVS(epoch) > errVS(epoch-1))
        y = true;
    end
end