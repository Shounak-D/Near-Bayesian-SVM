%% Code for Near-Bayesian SVM with OVA approach
% written by Shounak Datta, Junior Research Fellow, Indian Statistical Institute, Kolkata
% Version - 21st July 2015

%% program starts here

close all
% clear
% clc

if length(unique(y))==2
    NBSVM;
    return;
end

global rbf_sigma

C = input('Enter the cost of Regularization: ');
display('Positive kernel parameter runs the RBF kernel, negative parameter runs the Linear kernel...');
rbf_sigma = input('Enter the value of the kernel parameter: ');

%% k-fold cross-validation (multi-class)
k = 10; % number of partitions to be made
cls_num = length(unique(y));
sIze = zeros(1,cls_num);
leftover = zeros(1,cls_num);
for j = 1:cls_num
    sIze(j) = floor(size(x(:,y==j),2)/k);
    leftover(j) = size(x(:,y==j),2) - (k * floor(size(x(:,y==j),2)/k));
end
sIze = repmat(sIze,k,1);
flag = 0;
% sIze_idx = 1;
for j = 1:cls_num
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

accuracy = zeros(k,1);
svAcc = zeros(k,cls_num);

for i = 1:k
    %% extracting the multi-class training and test sets for a particular partition
    x_test = []; y_test = []; x_train = []; y_train = [];
    for j = 1:cls_num
        x_temp = xx(yy==j,:);
        x_temp = circshift(x_temp,-1*sIze_cum(i,j));
        y_temp = j * ones(size(x_temp,1),1);
        x_test = [x_test; x_temp(1:sIze(i,j),:)];
        y_test = [y_test; y_temp(1:sIze(i,j))];
        x_train = [x_train; x_temp((sIze(i,j)+1):end,:)];
        y_train = [y_train; y_temp((sIze(i,j)+1):end)];
    end
    Output = zeros(length(y_test), cls_num);
    for j = 1:cls_num
        %% training data and test data for a single classifier
        X = x_train;
        Y = ones(size(x_train,1),1);
        Y(y_train==j) = -1;
        X_test = x_test;
        Y_test = ones(size(x_test,1),1);
        Y_test(y_test==j) = -1;

        %% parameter setting
        smoOptions = svmsmoset;
        smoOptions.MaxIter = 70000; %20000;
        smoOptions.KKTViolationLevel = 0.25; %0.05;
        boxconstraint = ones(length(Y), 1);
        n1 = length(find(Y==-1));
        n2 = length(find(Y==1));
        c1 = C * length(Y) / n1;
        c2 = C * length(Y) / n2;
        boxconstraint(Y==-1) = c1;
        boxconstraint(Y==1) = c2;
%         boxconstraint = C * ones(length(Y), 1); % include to run C-SVM
        P = ones(length(Y), 1);
        p1 = n1 / length(Y);
        p2 = n2 / length(Y);
        P(Y==-1) = p1;
        P(Y==1) = p2;
%         P = ones(length(Y), 1); % include to run C-SVM

        %% solve by SMO
        [alphas, offset] = NBSVM_SMO(X, Y, boxconstraint, P, @rbf_kernel2, smoOptions);

        svIndex = find(alphas > sqrt(eps));
        svAcc(i,j) = length(svIndex)/length(X);
        output = [];
        data_indx = 1;
        while data_indx < length(X_test)
            X_test_part = X_test(data_indx: min(length(X_test), data_indx + 5000), :);
            data_indx = data_indx + 5001;

            KK = rbf_kernel2(X_test_part, X);
            output_part = sign( KK * (Y.*alphas) + offset);
            output = [output; output_part];
        end
        Output(:,j) = output;
        fprintf('Finished training Classifier %d.\n',j);
    end
    neg_count = (cls_num - sum(Output,2))/2; % counts the number of classifiers that claim each point to be in the target class
%     neg_count = 1./neg_count;
    ID_idx = zeros(length(y_test),1);
    for q = 1:length(ID_idx)
        if (Output(q,y_test(q))==-1)
            ID_idx(q) = 1; % finds whether the classifier corresponding to the actual class has classified correctly
        end
    end
    pointwise_acc = neg_count(find(ID_idx==1)); % finds the number of possible of assignments for each point that MAY be assigned to the actual class
    accuracy(i) = sum(1./pointwise_acc)/length(y_test);
    if accuracy(i) > 1
        error('Accuracy cannot be greater than one!');
    end
    fprintf('Finished validating over partition %d.\n',i);
end

avg_acc = mean(accuracy);
avg_svAcc = sum(sum(svAcc))/(size(svAcc,1)*size(svAcc,2));

%% plotting the performance

% plot( -1 : 0.01 : 1, accuracy_all)
% xlabel('tau')
% ylabel('accuracy')
% title('test accuracy for different tau (traversal algorithm)')
