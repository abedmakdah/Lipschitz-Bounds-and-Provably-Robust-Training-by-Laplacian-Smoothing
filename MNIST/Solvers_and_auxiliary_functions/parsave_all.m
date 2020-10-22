function parsave_all(fname, X_vertex, X_train, Y_train, X_test, Y_test...
    , G, Incidence, Indices_train_vertex, Indices_test_vertex)
  cd graphs
  save(fname, 'X_vertex', 'X_train', 'Y_train', 'X_test', 'Y_test', 'G',...
                'Incidence', 'Indices_train_vertex', 'Indices_test_vertex');
  cd ../
end