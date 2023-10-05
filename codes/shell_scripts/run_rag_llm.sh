MODEL_PATH="ollama"
VECTOR_DB_PATH="/data1/somank/llm_data/vectorDB/disease_context_chromaDB_using_pubmed_bert_sentence_transformer_model_with_chunk_size_650"
SENTENCE_EMBEDDING_MODEL="pritamdeka/S-PubMedBert-MS-MARCO"
QUESTION_TYPE="cmd_prompt"


python ../py_scripts/rag_with_explicit_retrieval.py $MODEL_PATH $VECTOR_DB_PATH $SENTENCE_EMBEDDING_MODEL $QUESTION_TYPE