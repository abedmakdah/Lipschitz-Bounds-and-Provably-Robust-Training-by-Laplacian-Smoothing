function data=load_graph_dataset(n_neighbors,N_vertex) %Kmeans

current_dir=pwd;
cd ../Graph_construction/Kmeans/graphs
data=load(sprintf('graph_Neigh%d_Vertex%d', [n_neighbors,N_vertex]));

cd (current_dir)

end

