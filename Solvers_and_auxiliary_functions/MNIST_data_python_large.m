function [X_train Y_train X_test Y_test] = MNIST_data_python_large


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loading MNIST dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
current_dir=pwd;
%cd ('../DATA')
load('XTest.mat')
load('YTest.mat')
load('XTrain.mat')
load('YTrain.mat')

N_train=size(X_train,1);
N_test=size(X_test,1);

for i=1:N_train
    
    X_train_aux(:,i)= reshape(flip(rot90(reshape(X_train(i,:),28,28),-1),2),[],1);
    
    if i<=N_test
    X_test_aux(:,i)=reshape(flip(rot90(reshape(X_test(i,:),28,28),-1),2),[],1);
    end
end

clear X_train X_test

X_train=X_train_aux;
X_test=X_test_aux;

Y_train=Y_train';
Y_test=Y_test';
% cd ../../tradeoff_adv_vs_nom_acc
%cd (current_dir)




end

