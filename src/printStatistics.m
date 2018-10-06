function printStatistics(confusionMatrix)
%PRINTSTATISTIC
%   *confusionMatrix: matrice di confusione ottenuta testando
%                     la rete addestrata su di un test set
    fprintf('Confusion Matrix:\n');
    fprintf('\t\t\tassigned class');
    for i=1:10
        fprintf('\t%d',(i-1));
    end
    fprintf('\ntrue class\n');
    for i=1:10
        fprintf('%d:\t\t\t\t\t\t\t',(i-1));
        for j=1:10
            fprintf('%d\t', confusionMatrix(i,j));
        end
        fprintf('\n');
    end
    for i=1:10
        sp = sum(confusionMatrix(:,i));
        if(sp == 0)
            precision = 0;
        else
            precision = confusionMatrix(i,i)/sp;
        end
        sr = sum(confusionMatrix(i,:));
        if(sr == 0)
            recall = 0;
        else
            recall = confusionMatrix(i,i)/sr;
        end
        fprintf('Precision for digit #%d: %f\n',(i-1), precision);
        fprintf('Recall for digit #%d: %f\n\n',(i-1), recall);
    end
    accuracy = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));
    fprintf('Accuracy: %f\n', accuracy);    
end