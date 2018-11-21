%Main 
%Tramite questo script e' possibile creare, allenare e 
%testare una rete feed forward fully connected sul
%dataset MNIST
%Si vedano i CONFIGURATION FLAGS per personalizzare
%la configurazione della rete, le dimensioni di
%trainig/validation/test set, l'algoritmo 
%di aggiornamento pesi desiderato

clc;
clear;

%CONFIGURATION FLAG
pca = true; %booleano per utilizzare o meno la PCA
numNodes = [100 10]; %descrizione del numero di nodi per layer della rete
functions = {@sigmoid, @identity}; %funzioni di output dei nodi una per strato
functionsDx = {@sigmoidDx, @identityDx}; %derivate funzioni di output dei nodi una per strato
learningType = 1; %0 per la resilient batch; 1 per gradientbatch
epochesGradientBatch = 300; %numero di epoche per il training batch con discesa del gradiente
epochesResilientBatch = 300; %numero di epoche per il training batch con RPROP
eta = 0.00025; %eta valido per gradientbatch e la prima epoca di RPROP
etaMinus = -0.6; %per RPROP
etaPlus = 1.05; %per RPROP
errorFunction = @crossEntropy; %funzione di errore 
errorFunctionDx = @crossEntropyDx; %derivata della funzione di errore 
afterProcessFunction = @softmax; %funzione di postProcessing dell'output della rete
stoppingCriterion = @progressQuotient; %criterio di stop per il training
onlyOneFunction = @onlyOne; 
suddivideParams = [15000 3000 3000]; %parametri di suddivisone del dataset per training/validation/test
suddivideCriterion = 1; %0 se suddivideParams sono relativi alla grandezza del dataset;
                        %1 se si riferiscono al numero esatto di elementi

images = loadMNISTImages('mnist/train-images.idx3-ubyte');
labels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');

%Suddivisione del dataset
tic;
if suddivideCriterion == 1
    [TrS, VaS, TeS, TrL, VaL, TeL] = suddivideDatasetAbsolute(images, labels, suddivideParams);
else
    [TrS, VaS, TeS, TrL, VaL, TeL] = suddivideDatasetRelative(images, labels, suddivideParams);
end
fprintf('Time for data partitioning: %.5f seconds\n', toc);

%Controllo se applicare la PCA
if pca
    tic;
    PC = myPca([TrS; VaS],99);
    TrS = TrS*PC;
    VaS = VaS*PC;
    TeS = TeS*PC;
    fprintf('Ho ridotto il num di features a %d\n', size(TrS, 2));
    fprintf('Time for data applying PCA: %.5f seconds\n', toc);
end

numFeatures = size(TrS, 2);

%Creazione rete
tic;
net = myNet(numFeatures, numNodes, functions, functionsDx);
fprintf('Time for network creation: %.5f seconds\n', toc);

fprintf('Numero strati: %d\nNumero nodi interni: ', size(numNodes, 1));
fprintf('%d ', numNodes(1));
fprintf('\n');
fprintf('Dimensione TrS: %d\nDimensione VaS: %d\nDimensione TeS: %d\n', size(TrS,1), size(VaS, 1), size(TeS, 1));
fprintf('Valore eta: %.6f\n', eta);
% Inizio la fase di learning mediante l'algortimo scelto col flag
if learningType == 1
    fprintf('Eseguo il gradientBatch\n');
    net = gradientBatch(net, TrS, TrL, VaS, VaL, epochesGradientBatch, errorFunction, errorFunctionDx, eta, afterProcessFunction);%, stoppingCriterion);
else
    fprintf('Eseguo il Resilient Batch\n');
    fprintf('Valore etaPlus: %.6f\nValore etaMinus: %.6f\n', etaPlus, etaMinus);
    net = rPropBatch(net, TrS, TrL, VaS, VaL, epochesResilientBatch, errorFunction, errorFunctionDx, eta, etaMinus, etaPlus, afterProcessFunction, stoppingCriterion);
end

%Valutazione delle performance della rete dopo il training sul test set
tic;
confusionMatrix = calculateStatistics(net, TeS, TeL, onlyOneFunction);
fprintf('Time for calculating the confusion matrix: %.5f seconds\n', toc);

printStatistics(confusionMatrix);