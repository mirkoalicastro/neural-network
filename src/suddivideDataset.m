function [TrS, VaS, TeS, TrL, VaL, TeL] = suddivideDataset(images, labels, percentages, length)
%SUDDIVIDEDATASET
%   *images: matrice delle immagini mnist
%   *labels: labels del dataset mnist
%   *percentages: array contenete il numero di elementi/frazione che il
%         training/validation/test set devono contenere
%   *length: 1 se percentages indica il numero assoluto di elementi per set
%            grandezza di MNISt se percentages indica il numero relativo di elementi per set
%   *TrS: training set con il quale addestrare la rete. E'una matrice
%         in cui gli elementi sono disposti su righe e dunque la colonna
%         j-esima rappresenta l'espressione della caratteristica j-esima
%         dello specifico elemento 
%   *VaS: validation set. E'una matrice
%         in cui gli elementi sono disposti su righe e dunque la colonna
%         j-esima rappresenta l'espressione della caratteristica j-esima
%         dello specifico elemento 
%   *TeS: test set sul quale testare la rete. E'una matrice
%         in cui gli elementi sono disposti su righe e dunque la colonna
%         j-esima rappresenta l'espressione della caratteristica j-esima
%         dello specifico elemento 
%   *TrL: insieme delle label degli elementi del training set. Alla riga
%         i-esima del TrS corrisponde la Label i-esima
%   *VaL: insieme delle label degli elementi del validation set. Alla riga
%         i-esima del TrS corrisponde la Label i-esima
%   *TeL: insieme delle label degli elementi del test set. Alla riga
%         i-esima del TrS corrisponde la Label i-esima


%Permette di suddividere il dataset MNIST il training/validation/test set
%fornendo in input un array in cui viene specificato il numero esatto di 
%elementi/frazione che ogni set deve contenere

    TrS = images(:,1:percentages(1)*length)';
    TrL = zeros(percentages(1)*length,10);
    k = 1;
    for i=1:(percentages(1)*length)
        TrL(k,:) = zeros(1,10);
        TrL(k, labels(i)+1) = 1;
        k = k + 1;
    end
    VaS = images(:,percentages(1)*length+1:percentages(1)*length+percentages(2)*length)';
    VaL = zeros(percentages(2)*length,10);
    k = 1;
    for i=i+1:((percentages(1)*length)+percentages(2)*length)
        VaL(k,:) = zeros(1,10);
        VaL(k, labels(i)+1) = 1;
        k = k + 1;
    end
    TeS = images(:,percentages(2)*length+percentages(1)*length+1:percentages(1)*length+percentages(2)*length+percentages(3)*length)';
    TeL = zeros(percentages(3)*length,10);
    k = 1;
    for i=i+1:(percentages(3)*length)+(percentages(2)*length)+(percentages(1)*length)
        TeL(k,:) = zeros(1,10);
        TeL(k, labels(i)+1) = 1;
        k = k + 1;
    end
    k = 1;
    return
end

