function net = gradientBatch(net, TrS, TrL, VaS, VaL, epoches, errorFunction, errorFunctionDx, eta, afterProcessFunction, stoppingCriterion)
%GRADIENTBATCH
%   *net: rete neurale feed forward fully connected
%   *TrS: training set con il quale addestrare la rete. E'una matrice
%         in cui gli elementi sono disposti su righe e dunque la colonna
%         j-esima rappresenta l'espressione della caratteristica j-esima
%         dello specifico elemento 
%   *TrL: insieme delle label degli elementi del training set. Alla riga
%         i-esima del TrS corrisponde la Label i-esima
%   *VaS: validation set sul quale viene calcolato l'errore di classificazione
%         ad ogni epoca e confrontato con quello sul training set
%         E'una matrice in cui gli elementi sono disposti su righe e dunque la colonna
%         j-esima rappresenta l'espressione della caratteristica j-esima
%         dello specifico elemento 
%   *VaL: insieme delle label degli elementi del validation set. Alla riga
%         i-esima del TrS corrisponde la Label i-esima
%   *epoches: numero massimo di epoche per l'addestramento
%   *errorFunction: funzione di errore da minimizzare
%   *errorFunctionDx: derivata della funezione di errore rispetto all'output
%   *eta: parametro attraverso il quale controllo l'aggiornamento
%         dei pesi/bias in congiunzione con la valutazione delle 
%         derivate della funzione di errore rispetto ai pesi
%   *afterProcessFunction: funzione da applicare all'uscita della rete
%   *stoppingCriterion: funzione per valutare se stoppare l'apprendimento
%                       prima della fine delel epoche
%   *net: net in input addestrata

%Riferimenti: lezioni frontali, Bishop

%Permette di allenare una rete costruita con myNet attraverso un aggiornamento 
%dei pesi in modalita' batch ottenuti tramite la discesa del gradiente


%Inizializzazione dei vettori per meemorizzare l'errore di classificazione
% per il training/validation set ad ogni epoca
    errTS = zeros(1, epoches);
    errVS = zeros(1, epoches);
    time = 0;

%Variabili ove memorizzare gli errori minimi ottenuti e la rete che
%ha ottenuto l'errore minore sul validation set e l'epoca in cui ho
%registrato questo errore minimo
    errMinTS = 0;
    errMinVS = 0;
    bestNet = net;
    bestEpoch = 0;


    for epoch=1:epoches
        tic;
%Ad ogni epoca forward prop, calcolo le derivate della funzione di errore rispetto
%ai pesi con la back prop e aggiorno i pesi con la discesa del gradiente
        [dW, dB] = backProp(net, TrS, TrL, errorFunctionDx, afterProcessFunction);
        net = gradientDescent(net, dW, dB, eta);

%Calcolo l'errore per questa epoca sul training/validation set
        errTS(epoch) = calculateErrorOn(net, TrS, TrL, errorFunction, afterProcessFunction)/size(TrS,1);
        errVS(epoch) = calculateErrorOn(net, VaS, VaL, errorFunction, afterProcessFunction)/size(VaS,1);

        if(epoch == 1 || errMinTS > errTS(epoch))
            errMinTS = errTS(epoch);
        end

%Controllo se ho registrato la migliore performance per salvare i parametri
        if(epoch == 1 || errMinVS > errVS(epoch))
            errMinVS = errVS(epoch);
            bestNet = net;
            bestEpoch = epoch;
        end
        
        time = time + toc;
        %fprintf('Errore sul TS all epoca #%d: %d\n', epoch, errTS(epoch));
        %fprintf('Errore sul VS all epoca #%d: %d\n\n', epoch, errVS(epoch));

%Se ho un un criterio di stop e mi dice di fermarmi termino
%all'epoca corrente l'addestramento della rete
        if exist('stoppingCriterion','var') && stoppingCriterion(errTS, errVS, epoch, errMinTS, errMinVS)
           fprintf('Mi fermo per il criterio di stop.\n');
           break;
        end
    end
    hold on;
    plot(errTS(1:epoch), 'b');
    plot(errVS(1:epoch), 'r');
    legend('TS','VS');
    title('GradientBatch');
    hold off;
    fprintf('Average time for one gradientBatch epoch: %.5f seconds\n', (time/epoches));
    fprintf('Time for all the gradientBatch epoches: %.5f seconds\n', time);
    net = bestNet;
    fprintf('la migliore e %d\n', bestEpoch);
end