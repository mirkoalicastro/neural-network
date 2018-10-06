function net = rPropBatch(net, TrS, TrL, VaS, VaL, epoches, errorFunction, errorFunctionDx, eta, etaMinus, etaPlus, afterProcessFunction, stoppingCriterion)
%RPROPBATCH
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
%   *etaMinus: parametro attraverso il quale rallento l'aggiornamento
%         dei pesi/bias in congiunzione con la valutazione delle 
%         derivate della funzione di errore rispetto ai pesi dell'epoca
%         attuale e precedente
%   *etaPlus:parametro attraverso il quale accellero l'aggiornamento
%         dei pesi/bias in congiunzione con la valutazione delle 
%         derivate della funzione di errore rispetto ai pesi dell'epoca
%         attuale e precedente
%   *afterProcessFunction: funzione da applicare all'uscita della rete
%   *stoppingCriterion: funzione per valutare se stoppare l'apprendimento
%                       prima della fine delel epoche
%   *net: net in input addestrata

%Riferimenti: lezioni frontali, Bishop

%La seguente funzione permette di addestrare una rete myNet utilizzando come
%criterio di aggiornamento dei pesi la resilient back propagation che permette, 
%studiando i segni delle derivate della funzione di errore rispetto ai pesi
%di due epoche consecutive, di controllare il movimento sulla funzione di errore
%accellerando quando si e' in fase di discesa dell'errore

%Inizializzazione dei vettori per meemorizzare l'errore di classificazione
% per il traning/validation set ad ogni epoca
    errTS = zeros(1,epoches);
    errVS = zeros(1,epoches);


    time = 0;
    tic;
%Per la prima epoca viene usata la discesa del gradiente per ricavare i primi scostamenti dei pesi
    [dWprev, dBprev] = backProp(net, TrS, TrL, errorFunctionDx, afterProcessFunction);
    [net, deltaWprev, deltaBprev] = gradientDescent(net, dWprev, dBprev, -eta);
    time = time + toc;
    errTS(1) = calculateErrorOn(net, TrS, TrL, errorFunction, afterProcessFunction)/size(TrS,1);
    errVS(1) = calculateErrorOn(net, VaS, VaL, errorFunction, afterProcessFunction)/size(VaS,1);

    %fprintf('Errore sul TS all epoca #1: %d\n', errTS(1));
    %fprintf('Errore sul VS all epoca #1: %d\n\n', errVS(1));

    errMinTS = errTS(1); 
    errMinVS = errVS(1); 

    for epoch=2:epoches
        tic;
        
%Propagazione in avanti gli input alla rete
        [dW, dB] = backProp(net, TrS, TrL, errorFunctionDx, afterProcessFunction);

%Inizializzazione delle strutture per mantenere gli scostamenti dei pesi
%per ogni epoca
        deltaW = cell(1, net.numLayers);
        deltaB = cell(1, net.numLayers);  
        for layer=1:net.numLayers
            
            deltaW{layer} = zeros(size(deltaWprev{layer},1), size(deltaWprev{layer},2));
            deltaB{layer} = zeros(size(deltaBprev{layer},1), size(deltaBprev{layer},2));

%Analisi delle derivate dei pesi per questa epoca e la precedente
            signdW = dWprev{layer} .* dW{layer};
            signdB = dBprev{layer} .* dB{layer};

%I pesi sono aggornati secondo quanto stabilito dalla Resilient back propagation
%con gli scostamenti  dati dallo scostamento precedente per etaPlus o etaMinus 
%a secondo che le derivate siano concordi o meno       

            deltaW{layer}(signdW>0) = deltaWprev{layer}(signdW>0) * etaPlus;
            deltaW{layer}(signdW==0) = deltaWprev{layer}(signdW==0);
            deltaW{layer}(signdW<0) = deltaWprev{layer}(signdW<0) * etaMinus;
            
            deltaB{layer}(signdB>0) = deltaBprev{layer}(signdB>0) * etaPlus;
            deltaB{layer}(signdB==0) = deltaBprev{layer}(signdB==0);
            deltaB{layer}(signdB<0) = deltaBprev{layer}(signdB<0) * etaMinus;
            
            net.W{layer} = net.W{layer} + deltaW{layer};
            net.B{layer} = net.B{layer} + deltaB{layer};            
        end
        time = time + toc;

%Storage dei valori attuali per il confronto per la prossima epoca   
        dWprev = dW;
        dBprev = dB;
        deltaWprev = deltaW;
        deltaBprev = deltaB;

        errTS(epoch) = calculateErrorOn(net, TrS, TrL, errorFunction, afterProcessFunction)/size(TrS,1);
        errVS(epoch) = calculateErrorOn(net, VaS, VaL, errorFunction, afterProcessFunction)/size(VaS,1);
        
        %fprintf('Errore sul TS all epoca #%d: %d\n', epoch, errTS(epoch));
        %fprintf('Errore sul VS all epoca #%d: %d\n\n', epoch, errVS(epoch)); 

        if(errMinTS > errTS(epoch))
            errMinTS = errTS(epoch); 
        end
        if(errMinVS > errVS(epoch))
            errMinVS = errVS(epoch); 
        end        
        
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
    title('ResilientBatch');
    hold off;
    fprintf('Average time for one resilientPropBatch epoch: %.5f seconds\n', (time/epoches));
    fprintf('Time for all the resilientPropBatch epoches: %.5f seconds\n', time);
    
end