function [TrS, VaS, TeS, TrL, VaL, TeL] = suddivideDatasetRelative(images, labels, percentages)
%SUDDIVIDEDATASETRELATIVE
%   *images: matrice delle immagini mnist
%   *labels: labels del dataset mnist
%   *percentages: array contenete  la frazione di elementi che
%         training/validation/test set devono contenere
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
%fornendo in input un array in cui viene specificato la frazione di 
%elementi che ogni set deve contenere rispetto alla grandezza di MNIST

    [TrS, VaS, TeS, TrL, VaL, TeL] = suddivideDataset(images, labels, percentages, size(labels,1));
end

