# train: ffn, residual, on codes
bash train/train_mapping_block.sh FFN True $1
bash train/map_embd_and_test_rag.sh FFN models/FFN_residual_True_on_codes.pth models/embeddings/QUERY_Test_FFN_residual_True_on_codes.pt train/test_rag/QUERY_Test_FFN${1}_residual_True_on_codes FFN_residual_True_on_codes True $1
# train: ffn, residual, on queries
bash train/train_mapping_block_on_query.sh FFN True $1
bash train/map_embd_and_test_rag.sh FFN models/FFN_residual_True_on_queries.pth models/embeddings/QUERY_Test_FFN_residual_True_on_queries.pt train/test_rag/QUERY_Test_FFN${1}_residual_True_on_queries FFN_residual_True_on_queries True $1
# train: ffn, no residual, on codes
bash train/train_mapping_block.sh FFN False $1
bash train/map_embd_and_test_rag.sh FFN models/FFN_residual_False_on_codes.pth models/embeddings/QUERY_Test_FFN_residual_False_on_codes.pt train/test_rag/QUERY_Test_FFN${1}_residual_False_on_codes FFN_residual_False_on_codes False $1
# train: ffn, no residual, on queries
bash train/train_mapping_block_on_query.sh FFN False $1
bash train/map_embd_and_test_rag.sh FFN models/FFN_residual_False_on_queries.pth models/embeddings/QUERY_Test_FFN_residual_False_on_queries.pt train/test_rag/QUERY_Test_FFN${1}_residual_False_on_queries FFN_residual_False_on_queries False $1