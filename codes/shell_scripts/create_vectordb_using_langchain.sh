DATA_PATH="/data1/somank/llm_data/spoke_data/disease_with_relation_to_genes.pickle"
CHUNK_SIZE=650
CHUNK_OVERLAP=200
BATCH_SIZE=200
SENTENCE_EMBEDDING_MODEL="pritamdeka/S-PubMedBert-MS-MARCO"
VECTOR_DB_NAME="/data1/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_pubmed_bert_sentence_transformer_model_with_chunk_size_650"

start_time=$(date +%s)
python ../py_scripts/vectorDB/create_vectordb_using_langchain.py $DATA_PATH $CHUNK_SIZE $CHUNK_OVERLAP $BATCH_SIZE $SENTENCE_EMBEDDING_MODEL $VECTOR_DB_NAME
wait
end_time=$(date +%s)
time_taken_hours=$(( (end_time - start_time) / 3600 ))
echo "Time taken to complete : $time_taken_hours hours"