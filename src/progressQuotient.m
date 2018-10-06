function [y, PQ] = progressQuotient(errTS, errVS, epoch, minErrTS, minErrVS, threshold)
%PROGRESSQUOTIENT
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
%   *PQ: quoziente di perdita di generalizzazione e progresso

% Riferimenti: lezioni frontali, paper Early Stopping | but when? pg.5


%Permette di fermare l'apprendimento della rete quando dopo una strip
%qualche strip di k epoche ove GL(epoca)/(quanto la media dell'errore
%nella strip e' grande rispetto al minimo errore nella soglia)> soglia


    y = false;

%Se non passo la soglia dall'esterno uso questa soglia di fallback

    if ~exist('threshold','var')
        % 0.25 gradientBatch senza PCA
        % 0.30 gradientBatch con PCA
        threshold = 0.005;
    end

%Grandezza strip
    k = 24;

%Aggiungo questo valore per evitare una divisione per 0
    smooth = 0.000001;
    if(epoch < k)
        return
    end
    
    GG = sum(errTS(max(1,epoch-k+1):epoch))/((k*min(errTS(max(1,epoch-k+1):epoch)))+smooth) - 1;
    [~, GL] = generalizationLoss(errTS, errVS, epoch, minErrTS, minErrVS, threshold);
    PQ = GL / (GG*10);
    if(PQ > threshold)
        y = true;
    end
end

