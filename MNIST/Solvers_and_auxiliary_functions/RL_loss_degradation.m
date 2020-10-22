function [max_loss_deg,neigbor_max_loss_deg]=...
    RL_loss_degradation...
    (X_vertex, Minimizer, Delta_norm)


% Generate and test accuracy on adversarial set

F=Minimizer;
N_vertex=size(X_vertex,2);

max_deg=zeros(1,N_vertex);
neigbor_max_deg=zeros(1,N_vertex);

for r = 1 : N_vertex
    neighbors_indices=[];
    sample=X_vertex(:,r);
    
    neighbors_indices=find(vecnorm(X_vertex-sample,2)<=0.007); %& vecnorm(X_vertex-sample,2)~=0
    
    if ~isempty(neighbors_indices)
        %r
        
        f_error=F(:,r)-F(:,neighbors_indices);
        loss_degradation=vecnorm(f_error,2);
        [max_loss_deg_per_vertex, index_aux]=max(loss_degradation);
        max_deg(r)=max_loss_deg_per_vertex;
        neigbor_max_deg(r)=neighbors_indices(index_aux);
        
    end
    
end

[max_loss_deg, vertex_max_loss_deg]=max(max_deg);
neigbor_max_loss_deg=neigbor_max_deg(vertex_max_loss_deg);

end
