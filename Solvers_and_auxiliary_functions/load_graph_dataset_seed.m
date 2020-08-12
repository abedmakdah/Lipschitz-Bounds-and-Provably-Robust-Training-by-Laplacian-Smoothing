function data=load_graph_dataset_seed(n_neighbors,N_vertex,seed,method)


current_dir=pwd;
if method==0
    cd ../../Graph_construction/random
else
    cd ../../Graph_construction/Kmeans
end

cd(sprintf('rng_%d/graphs', seed))
data=load(sprintf('graph_Neigh%d_Vertex%d', [n_neighbors,N_vertex]));

cd (current_dir)

end

