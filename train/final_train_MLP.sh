# train: mlp, residual, on codes
bash train/train_mapping_block.sh MLP True
bash train/map_embd_and_test_rag.sh MLP models/MLP_residual_True_on_codes.pth models/embeddings/QUERY_Test_MLP_residual_True_on_codes.pt train/test_rag/QUERY_Test_MLP_residual_True_on_codes MLP_residual_True_on_codes True
# train: mlp, residual, on queries
bash train/train_mapping_block_on_query.sh MLP True
bash train/map_embd_and_test_rag.sh MLP models/MLP_residual_True_on_queries.pth models/embeddings/QUERY_Test_MLP_residual_True_on_queries.pt train/test_rag/QUERY_Test_MLP_residual_True_on_queries MLP_residual_True_on_queries True
# train: mlp, no residual, on codes
bash train/train_mapping_block.sh MLP False
bash train/map_embd_and_test_rag.sh MLP models/MLP_residual_False_on_codes.pth models/embeddings/QUERY_Test_MLP_residual_False_on_codes.pt train/test_rag/QUERY_Test_MLP_residual_False_on_codes MLP_residual_False_on_codes False
# train: mlp, no residual, on queries
bash train/train_mapping_block_on_query.sh MLP False
bash train/map_embd_and_test_rag.sh MLP models/MLP_residual_False_on_queries.pth models/embeddings/QUERY_Test_MLP_residual_False_on_queries.pt train/test_rag/QUERY_Test_MLP_residual_False_on_queries MLP_residual_False_on_queries False
