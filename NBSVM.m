%% Code for Near-Bayesian SVM
% written by Shounak Datta, Junior Research Fellow, Indian Statistical Institute, Kolkata
% Version - 21st July 2015

%% program starts here

close all
% clear
% clc

global rbf_sigma C

% if isempty(C)
    C = input('Enter the cost of Regularization: ');
% end
% if isempty(rbf_sigma)
    display('Positive kernel parameter runs the RBF kernel, negative parameter runs the Linear kernel...');
    rbf_sigma = input('Enter the value of the kernel parameter: ');
% end

%% k-fold cross-validation
k = 10; % number of partitions to be made
% sIze = floor(length(x)/k) * ones(k,1);
% leftover = length(x) - (k * floor(length(x)/k));
% sIze_idx = 1;
% while leftover > 0
%     sIze(sIze_idx) = sIze(sIze_idx) + 1;
%     leftover = leftover - 1;
%     sIze_idx = sIze_idx + 1;
% end

sIze(1) = floor(size(x(:,y==-1),2)/k);
sIze(2) = floor(size(x(:,y==1),2)/k);
leftover(1) = size(x(:,y==-1),2) - (k * floor(size(x(:,y==-1),2)/k));
leftover(2) = size(x(:,y==1),2) - (k * floor(size(x(:,y==1),2)/k));
sIze = repmat(sIze,k,1);
flag = 0;
for j = 1:2
    if flag==0
        sIze_idx = 1;
    else
        sIze_idx = k;
    end
    while leftover(j) > 0
        sIze(sIze_idx,j) = sIze(sIze_idx,j) + 1;
        leftover(j) = leftover(j) - 1;
        if flag==0
            sIze_idx = sIze_idx + 1;
        else
            sIze_idx = sIze_idx - 1;
        end
    end
    flag = ~flag;
end
sIze_cum = cumsum(sIze);
sIze_cum = circshift(sIze_cum,1);

rand_IDX = randperm(length(x));
xx = x(:,rand_IDX)';
yy = y(rand_IDX)';
yy = (yy + 3)./2; %turn labels -1 and 1 into 1 and 2, resp.

accuracy = ones(k,1);
svAcc = zeros(k,1);

for i = 1:k
    %% extracting the multi-class training and test sets for a particular partition
    x_test = []; y_test = []; x_train = []; y_train = [];
    for j = 1:2
        x_temp = xx(yy==j,:);
        x_temp = circshift(x_temp,-1*sIze_cum(i,j));
        y_temp = j * ones(size(x_temp,1),1);
        x_test = [x_test; x_temp(1:sIze(i,j),:)];
        y_test = [y_test; y_temp(1:sIze(i,j))];
        x_train = [x_train; x_temp((sIze(i,j)+1):end,:)];
        y_train = [y_train; y_temp((sIze(i,j)+1):end)];
    end
    y_train = y_train.*2 - 3; %turn labels 1 and 2 back to -1 and 1, resp.
    y_test = y_test.*2 - 3; %turn labels 1 and 2 back to -1 and 1, resp.
    
%     x_test = xx(1:sIze(i),:);
%     y_test = yy(1:sIze(i));
%     x_train = xx((sIze(i)+1):end,:);
%     y_train = yy((sIze(i)+1):end);
%     xx = circshift(xx,-1*sIze(i));
%     yy = circshift(yy,-1*sIze(i));

    %% training data and test data
    X = x_train;
%     Xtrain{i} = X;
    Y = y_train;
%     Ytrain{i} = Y;
    X_test = x_test;
    Y_test = y_test;

    %% parameter setting
    smoOptions = svmsmoset;
    smoOptions.MaxIter = 25000;
    smoOptions.KKTViolationLevel = 0.05;
    boxconstraint = ones(length(Y), 1);
    n1 = length(find(Y==-1));
    n2 = length(find(Y==1));
    c1 = C * length(Y) / n1;
    c2 = C * length(Y) / n2;
    boxconstraint(Y==-1) = c1;
    boxconstraint(Y==1) = c2;
%     boxconstraint = C * ones(length(Y), 1); % include to run C-SVM
    P = ones(length(Y), 1);
    p1 = n1 / length(Y);
    p2 = n2 / length(Y);
    P(Y==-1) = p1;
    P(Y==1) = p2;
%     P = ones(length(Y), 1); % include to run C-SVM

    %% solve by SMO
    [alphas, offset] = NBSVM_SMO(X, Y, boxconstraint, P, @rbf_kernel2, smoOptions);
%     Alph(i,:) = [alphas', offset];

    svIndex = find(alphas > sqrt(eps));
    svAcc(i) = length(svIndex)/length(X);
%     accuracy(i) = 1;
%     output = [];
    data_indx = 1;
    while data_indx < length(Y_test)
        X_test_part = X_test(data_indx: min(length(Y_test), data_indx + 5000), :);
        Y_test_part = Y_test(data_indx: min(length(Y_test), data_indx + 5000));
        data_indx = data_indx + 5001;

        KK = rbf_kernel2(X_test_part, X);
        output_part = sign( KK * (Y.*alphas) + offset);
        %output = [output; output_part];
        accuracy(i) = accuracy(i) - sum(abs(output_part - Y_test_part))/2/length(Y_test);
    end
    fprintf('Finished validating over partition %d.\n',i);
end

avg_acc = mean(accuracy);
avg_svAcc = mean(svAcc);

%% plotting the performance

% plot( -1 : 0.01 : 1, accuracy_all)
% xlabel('tau')
% ylabel('accuracy')
% title('test accuracy for different tau (traversal algorithm)')
