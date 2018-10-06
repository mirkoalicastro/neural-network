function [PC] = myPca(x, threshold)
%MYPCA
%   *x: set di valori sul quale effettuare la PCA
%   *threashold: soglia in percentuale di varianza richiesta
%   *PC: matrice degli m autovettori necessari per ridurre i set

%Riferimenti: Bishop, http://www.math.union.edu/~jaureguj/PCA.pdf,
% https://www.cs.princeton.edu/picasso/mats/PCA-Tutorial-Intuition_jp.pdf


%myPca ritorna la matrice contenete gli m autovalori legati 
%agli autovalori piu' grandi per trasformare gli input da uno 
%spazio n-dimensionale ad uno m-dimensionale con m<n


%Calcolo della media per ogni caratteristica e della matrice di covarianza
xMean=mean(x,1);
covariance=(x-xMean)'*(x-xMean);

%Estraggo le matrici di autovettori e autovalori
[P, V] = eig(covariance);

%GLi autovalori sono disposti sulla diagonale principale
eigenValues=diag(V);

%Riordino autovettori e a autovalori in ordine decrescente
[eigenValues,permut]=sort(eigenValues,'descend');
P=P(:,permut);

%Calcolo la somma cumulata della varianza e cerco 
%il primo valore >= alla soglia specificata
retainedVariance=cumsum(eigenValues)/sum(eigenValues);
threshold=threshold/100;
result=find(retainedVariance>=threshold);
dim=result(1);

%Restituisco solo i primi m autovettori piu' grandi
PC=P(:,1:dim);
return;
end

