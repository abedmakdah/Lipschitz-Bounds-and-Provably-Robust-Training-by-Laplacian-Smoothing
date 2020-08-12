function Y = primal_dual_dynamics_robust_learning(t, x, N_vertex, N_labels, X_vertex, L_max, Y_train, Indices_train_vertex, G, Incidence, N_train)
    
    f = x(1:N_labels*N_vertex);
    f = reshape(f,N_labels,N_vertex);
    
    Edges = x(N_labels*N_vertex+1:end);

    % Update value of edges
    TOL = 1e-10;
    
    Edges_new = (1e0)*(vecnorm(f*Incidence,2,1) - L_max * vecnorm(X_vertex*Incidence,2,1));
    
    aux = Edges <= TOL;
    aux_negative = Edges_new <= TOL;
    indices = find(aux & aux_negative' == 1);
    if ~isempty(indices)
        Edges_new(indices) = Edges_new(indices)*0;
    end
    % Edges(indices) = Edges(indices)*0; % Vishaal
    
    aux = adjacency(G,Edges);
    L = diag(sum(aux,2)) - aux;



    
    % Compute loss function
    f_X_train = f(:,Indices_train_vertex); % predicted value at each training point
    error = f_X_train - Y_train;
    
    Q = zeros(N_labels,N_vertex);
    
    [xx, yy] = ndgrid(Indices_train_vertex,1:size(error, 1));
    totals = accumarray([yy(:) xx(:)], reshape(error', 1, []));
    
    Q = totals/N_train;%./N_samples_per_vertex;
    
    % Update function
    x = -(L*f' + Q')';
    
    % Return values
    Y = [reshape(x,N_labels*N_vertex,1) ; Edges_new'];
    
end