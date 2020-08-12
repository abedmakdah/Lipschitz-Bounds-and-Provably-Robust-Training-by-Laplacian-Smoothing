function [DELTA, Accuracy_nominal, Accuracy_adversarial]=...
    RL_adversarial_perturbation_loss_degredation...
    (X_vertex, Indices_test_vertex, Minimizer,...
    X_test, Y_test, Delta_norm)


% Generate and test accuracy on adversarial set

F=Minimizer;
N_adv_test = size(Y_test,2);



%
index_test=Indices_test_vertex;

Delta = zeros(size(X_test));

parfor r = 1 : N_adv_test
    if r==88
        r
    end
    % Select a random data sample
    sample = X_test(:,r);
    label = Y_test(:,r);
    
    closest_vertex=X_vertex(:,index_test(r));
    [~, predicted_class] = max(F(:,index_test(r)));
    [~, true_class] = max(Y_test(:,r));
    
    %     %     f_x_pred=F(:,index_test(r));
    f_true=Y_test(:,r);
    
    %     if predicted_class == true_class
    
    
    z_prime=(X_vertex+closest_vertex)/2;
    
    aux=(X_vertex-closest_vertex)./vecnorm(X_vertex-closest_vertex);
    p_prime=closest_vertex + (((sample-closest_vertex)')*aux).*aux;
    
    inf_dist=vecnorm(z_prime-p_prime,inf);
    
    % %         %distance between vertex and closes vertex (might be zero)
    % %         % D=vecnorm(X_vertex-closest_vertex)/2;
    % %         % P=((sample-closest_vertex)')*(X_vertex-closest_vertex)./(2*D); %projection
    
    possible_target=find(inf_dist<=Delta_norm); %indices of the possible targets
    
    if ~isempty(possible_target)
        [~,possible_class]= max(F(:,possible_target));
        
        
        
        % indices within the desired distance and have different class
        %         aux_index = find(abs(possible_class-true_class)>=10^-4);
        
        %         if ~isempty(aux_index)
        f_pred_target=F(:,possible_target);
        
        f_error=f_pred_target-f_true;
        norm_f_error=sqrt(diag(f_error'*f_error));
        
        [target_error,p]=max(norm_f_error);
        
        
        
        dir_perturbation=(X_vertex(:,possible_target(p))-sample)...
            /norm((X_vertex(:,possible_target(p))-sample),inf); %direction of the perturbation
        perturbed_point=sample+ Delta_norm*dir_perturbation;
        
        Delta(:,r)=Delta_norm*dir_perturbation;
    end
    %         end
    
    % %         %         [~,target_class]=max(F(:, knnsearch(X_vertex', perturbed_point')));
    % %
    % %
    % %
    % %         %         p = 0;
    % %         %         flag = 1;
    % %         %         target_class = [];
    % %         %         while p < length(aux_index) && flag
    % %         %             p = p + 1;
    % %         %             %perturbation along the parallel diraction
    % %         %             dir_perturbation=(X_vertex(:,possible_target(aux_index(p)))-sample)...
    % %         %                 /norm((X_vertex(:,possible_target(aux_index(p)))-sample),inf); %direction of the perturbation
    % %         %             perturbed_point=sample+ Delta_norm*dir_perturbation;
    % %         %
    % %         %             [~,target_class]=max(F(:, knnsearch(X_vertex', perturbed_point')));
    % %         %             if abs(target_class-true_class)>=10^-4
    % %         %                 %Found target vertex
    % %         %                 %                     target_class = X_vertex(:,aux_index(p));
    % %         %                 flag = 0;
    % %         %                 Delta(:,r)=Delta_norm*dir_perturbation;
    % %         %             end
    % %         %
    % %         %
    % %         %         end
    
    %     end
end


% % Adv_test = adv_perturbations(X_test, Y_test, N_adv_test, Delta_norm, @predictor);

[~, true_class_test] = max(Y_test, [], 1);

[~, predicted_class_test] = max(F(:, knnsearch(X_vertex', X_test')), [], 1);

Accuracy_nominal = length(find(abs(predicted_class_test - true_class_test) <= 10^-4))...
    /size(Y_test,2);

[~, predicted_class_adv] = max(F(:, knnsearch(X_vertex', (X_test+Delta)')), [], 1);

Accuracy_adversarial = length(find(abs(predicted_class_adv - true_class_test) <= 10^-4))...
    /size(Y_test,2);

DELTA=Delta;
end
