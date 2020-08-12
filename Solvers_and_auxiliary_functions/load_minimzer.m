function F=load_minimzer(n_neighbors,N_vertex,T,L_max)

current_dir=pwd;
cd ('../Trained_minimizer')
cd(sprintf('Vertex%d_Neigh%d_Kmeans_T%d', [N_vertex,n_neighbors,T]))

F=load(sprintf('F_L%d', L_max));

cd (current_dir)

end

