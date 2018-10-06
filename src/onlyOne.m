function y = onlyOne(z)
%ONLYONE 
%   *z:matrice di valori
%   *y: matrice di valori



%La seguente funzione permette di converitre z
% in una mtrice y ove per ogni riga si avra' 1
% in corrispondenza del massimo di riga e 0 per 
%tutti gli altri valori

%Salvo gli indici del massimo per ogni riga
    [~, loc] = max(z,[],2); %equivalente a max(z')

%Sostituisco con tutti 0
    y = zeros(size(z, 1), size(z, 2));

%Ripristino 1 ove prima vi era il massimo di riga
    for i=1:size(z,1)
        y(i,loc(i)) = 1;
    end
    % e' inutile che faccio il softmax
end

