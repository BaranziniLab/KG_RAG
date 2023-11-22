VECTOR_DB_PATH="/data1/somank/llm_data/vectorDB/disease_context_chromaDB_using_pubmed_bert_sentence_transformer_model_with_chunk_size_650"
SENTENCE_EMBEDDING_MODEL="pritamdeka/S-PubMedBert-MS-MARCO"
QUESTION_DATA_PATH="/data1/somank/llm_data/spoke_data/test_questions_for_retrieval_performance.csv"
SAVE_DATA_PATH="/data1/somank/llm_data/analysis"

start_time=$(date +%s)
python ../py_scripts/estimation_of_retrieval_score_for_sentence_embedding_model.py $VECTOR_DB_PATH $SENTENCE_EMBEDDING_MODEL $QUESTION_DATA_PATH $SAVE_DATA_PATH
wait
end_time=$(date +%s)
time_taken_hours=$(( (end_time - start_time) / 3600 ))
echo "Time taken to complete : $time_taken_hours hours"