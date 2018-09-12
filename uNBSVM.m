%% Code for Near-Bayesian SVM with Unequal Costs
% written by Shounak Datta, Junior Research Fellow, Indian Statistical Institute, Kolkata
% Version - 21st July 2015

%% program starts here

close all
% clear
% clc

global rbf_sigma C w1 w2

% if isempty(C)
    C = input('Enter the cost of Regularization: ');
% end
% if isempty(w1)
    w1 = input('Enter the penalty for misclassification from the positive (target) class: ');
    w2 = input('Enter the penalty for misclassification from the negative class: ');
% end
% if isempty(rbf_sigma)
    display('Positive kernel parameter runs the RBF kernel, negative parameter runs the Linear kernel...');
    rbf_sigma = input('Enter the value of the kernel parameter: ');
% end

%% run other algorithms
% C_SVM;
% display('----------------------------------------------------------------');
% SDC_SVM;
% display('----------------------------------------------------------------');

%% k-fold cross-validation
k = 10; % number of partitions to be made
sIze = floor(length(x)/k) * ones(k,1);
leftover = length(x) - (k * floor(length(x)/k));
sIze_idx = 1;
while leftover > 0
    sIze(sIze_idx) = sIze(sIze_idx) + 1;
    leftover = leftover - 1;
    sIze_idx = sIze_idx + 1;
end
rand_IDX = randperm(length(x));
xx = x(:,rand_IDX)';
yy = y(rand_IDX)';
accuracy = ones(k,1);
precision = zeros(k,1);
sensitivity = zeros(k,1);
specificity = zeros(k,1);
gmeans = zeros(k,1);
svAcc = zeros(k,1);

for i = 1:k
    x_test = xx(1:sIze(i),:);
    y_test = yy(1:sIze(i));
    x_train = xx((sIze(i)+1):end,:);
    y_train = yy((sIze(i)+1):end);
    xx = circshift(xx,-1*sIze(i));
    yy = circshift(yy,-1*sIze(i));

    %% training data and test data
    X = x_train;
%     Xtrain{i} = X;
    Y = y_train;
%     Ytrain{i} = Y;
    X_test = x_test;
    Y_test = y_test;

    %% parameter setting
    smoOptions = svmsmoset;
    smoOptions.MaxIter = 150000; %25000;
    smoOptions.KKTViolationLevel = 0.1; %0.05;
    boxconstraint = ones(length(Y), 1);
    n1 = length(find(Y==-1));
    n2 = length(find(Y==1));
    c1 = C * length(Y) / n1;
    c2 = C * length(Y) / n2;
    boxconstraint(Y==-1) = c1;
    boxconstraint(Y==1) = c2;
%     boxconstraint = C * ones(length(Y), 1); % include to run C-SVM
    P = ones(length(Y), 1);
    p1 = n1 * w1 / length(Y);
    p2 = n2 * w2 / length(Y);
%     p1 = p1 / (p1 + p2);
%     p2 = 1 - p1;
    P(Y==-1) = p1 / (p1 + p2);
    P(Y==1) = p2 / (p1 + p2);
%     P = ones(length(Y), 1); % include to run C-SVM

    %% solve by SMO
    [alphas, offset] = NBSVM_SMO(X, Y, boxconstraint, P, @rbf_kernel2, smoOptions);
%     Alph(i,:) = [alphas', offset];

    svIndex = find(alphas > sqrt(eps));
    svAcc(i) = length(svIndex)/length(X);
%     accuracy(i) = 1;
    output = [];
    data_indx = 1;
    while data_indx < length(Y_test)
        X_test_part = X_test(data_indx: min(length(Y_test), data_indx + 5000), :);
        Y_test_part = Y_test(data_indx: min(length(Y_test), data_indx + 5000));
        data_indx = data_indx + 5001;

        KK = rbf_kernel2(X_test_part, X);
        output_part = sign( KK * (Y.*alphas) + offset);
        output = [output; output_part];
        accuracy(i) = accuracy(i) - sum(abs(output_part - Y_test_part))/2/length(Y_test);
    end
    posMask = (Y_test==-1); negMask = ~posMask;
    out_posMask = (output==-1); out_negMask = ~out_posMask;
    tp = sum(posMask.*out_posMask);
    fp = sum(negMask.*out_posMask);
    tn = sum(negMask.*out_negMask);
    fn = sum(posMask.*out_negMask);
    precision(i) = tp/(tp + fp);
    sensitivity(i) = tp/(tp + fn);
    specificity(i) = tn/(tn + fp);
    gmeans(i) = sqrt( sensitivity(i) * specificity(i) );
    fprintf('Finished validating uNBSVM over partition %d.\n',i);
end

avg_acc = mean(accuracy);
avg_prec = mean(precision);
avg_sens = mean(sensitivity);
avg_spec = mean(specificity);
avg_gmeans = mean(gmeans);
avg_svAcc = mean(svAcc);

% clearvars -except num_n C w1 w2 rbf_sigma x y... 
%     avg_acc avg_prec avg_sens avg_spec avg_gmeans avg_svAcc...
%     avg_sdcacc avg_sdcprec avg_sdcsens avg_sdcspec avg_sdcgmeans avg_sdcsvAcc...
%     avg_svmacc avg_svmprec avg_svmsens avg_svmspec avg_svmgmeans avg_svmsvAcc

%% plotting the performance

% plot( -1 : 0.01 : 1, accuracy_all)
% xlabel('tau')
% ylabel('accuracy')
% title('test accuracy for different tau (traversal algorithm)')
