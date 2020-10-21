 % Computes Robustness vs performance Tradeoff for MNIST

% Clear workspace and figures
clear all
close all
% clc
addpath ../Solvers_and_auxiliary_functions
addpath ../DATA

tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_vertex = 5000;
L_max = 500;
n_neighbors = 5;

T = 1000; %Time horizon for primal dual algorithm


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MNIST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[X_train Y_train X_test Y_test] = MNIST_data_python_large;

N_train = size(X_train,2);
N_test = size(X_test,2);
N_labels = size(Y_test,1);

% Normalize the input data to be within a hypercube of size 1
X_train=X_train./size(X_train,1);
X_test=X_test./size(X_test,1);

rng(8) %fix random generator seed

% Vertices (Kmeans)
% [IDX, X_vertex] = kmeans(X_train', N_vertex);
% X_vertex = X_vertex';

% Select vertices from training set
Indices = randperm(N_train);
X_vertex = X_train(:,Indices(1:N_vertex));

Indices_train_vertex = knnsearch(X_vertex', X_train')';

% Reomove vertices with no training points
[indices_sort Index_sorting] = sort(Indices_train_vertex);
N_samples_per_vertex = histcounts(indices_sort,(0:N_vertex)+.1);
aux = find(N_samples_per_vertex > 1e-5);
X_vertex = X_vertex(:,aux);
N_vertex = size(X_vertex,2);

Indices_train_vertex = knnsearch(X_vertex', X_train')';
Indices_test_vertex = knnsearch(X_vertex', X_test')';

% Reorder training points
[indices_sort Index_sorting] = sort(Indices_train_vertex);
X_train = X_train(:,Index_sorting);
Y_train = Y_train(:,Index_sorting);
Indices_train_vertex = indices_sort;
N_samples_per_vertex = histcounts(indices_sort,(0:N_vertex)+.1);

% This part is a sanity check. It can be removed.
if ~isempty(find(N_samples_per_vertex==0))
    disp('Not enough training samples per vertex')
    return
end

% Reorder testing points
[indices_sort Index_sorting] = sort(Indices_test_vertex);
X_test = X_test(:,Index_sorting);
Y_test = Y_test(:,Index_sorting);
Indices_test_vertex = indices_sort;

Neighbors = knnsearch(X_vertex',X_vertex','K',n_neighbors+1);

Neighbors = [kron(ones(1,n_neighbors),Neighbors(:,1)');reshape(Neighbors(:,2:end),N_vertex*n_neighbors,1)'];

G = digraph;
G = addnode(G,N_vertex);
G = addedge(G,Neighbors(1,:),Neighbors(2,:));

% Make graph symmetric
aux = (adjacency(G)+adjacency(G)')>0;
G = graph(aux);
Incidence = incidence(G);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Approximation_error = [];
Minimizer = {};
Accuracy_train = [];
Accuracy_test = [];
Lipschitz_num = [];

[confidence_train true_class_train] = max(Y_train, [], 1);
[confidence_train true_class_test] = max(Y_test, [], 1);

x0 = [zeros(N_labels*N_vertex,1) ; zeros(numedges(G),1)];

[Time X] = ode45(@(t,x) primal_dual_dynamics_robust_learning_noglobal(t, x,  N_vertex, N_labels,...
    X_vertex, L_max, Y_train, Indices_train_vertex, G, Incidence, N_train), [0 T], x0);

F = X(end,1:N_labels*N_vertex)';
F = reshape(F,N_labels,N_vertex);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performance metrics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Numerical Lipschitz
Lipschitz_num=max(abs(vecnorm(F*Incidence)./vecnorm(X_vertex*Incidence)));

% Evaluate performance on testing set
f_X_train = F(:,Indices_train_vertex);
Approximation_error_train = norm(f_X_train - Y_train,'fro')/N_train;

f_X_test = F(:,Indices_test_vertex);
Approximation_error_test = norm(f_X_test - Y_test,'fro')/N_test;

F = F./vecnorm(F,2,1);

f_X_train = F(:,Indices_train_vertex);
[confidence_train predicted_class_train] = max(f_X_train, [], 1);

Accuracy_train = length(find(abs(predicted_class_train - true_class_train) <= 10^-4))/N_train

f_X_test = F(:,Indices_test_vertex);
[confidence_train predicted_class_test] = max(f_X_test, [], 1);

Accuracy_test = length(find(abs(predicted_class_test - true_class_test) <= 10^-4))/N_test

max_confidence = mean(max(F))  %confidence at each lipstchitz bound

toc