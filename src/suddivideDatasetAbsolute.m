function [TrS, VaS, TeS, TrL, VaL, TeL] = suddivideDatasetAbsolute(images, labels, num)
%SUDDIVIDEDATASETABSOLUTE
%   *images: matrice delle immagini mnist
%   *labels: labels del dataset mnist
%   *num: array contenete il numero di elementi che il
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
%fornendo in input un array in cui viene specificato il numero esatto di 
%elementi che ogni set deve contenere


    [TrS, VaS, TeS, TrL, VaL, TeL] = suddivideDataset(images, labels, num, 1);
end

