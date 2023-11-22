VECTOR_DB_PATH="/data1/somank/llm_data/vectorDB/disease_context_chromaDB_using_pubmed_bert_sentence_transformer_model_with_chunk_size_650"
SENTENCE_EMBEDDING_MODEL="pritamdeka/S-PubMedBert-MS-MARCO"
MODEL_NAME="TheBloke/Llama-2-13b-Chat-GPTQ"
BRANCH_NAME="gptq-4bit-64g-actorder_True"
CACHE_DIR="/data1/somank/llm_data/llm_models/huggingface"

python ../py_scripts/rag_based_text_generation/rag_with_retrieval_threshold_HuggingFace.py $VECTOR_DB_PATH $SENTENCE_EMBEDDING_MODEL $MODEL_NAME $BRANCH_NAME $CACHE_DIR