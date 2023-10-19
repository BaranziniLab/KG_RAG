VECTOR_DB_PATH="/data/somank/llm_data/vectorDB/disease_context_chromaDB_using_pubmed_bert_sentence_transformer_model_with_chunk_size_650"
SENTENCE_EMBEDDING_MODEL="pritamdeka/S-PubMedBert-MS-MARCO"
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
BRANCH_NAME="main"
QUESTION_PATH="/data/somank/llm_data/analysis/test_questions.csv"
SAVE_PATH="/data/somank/llm_data/analysis"
CACHE_DIR="/data/somank/llm_data/llm_models/huggingface"

python ../py_scripts/rag_based_text_generation/Llama/rag_with_retrieval_threshold_for_true_false_generation_Llama_HuggingFace.py $VECTOR_DB_PATH $SENTENCE_EMBEDDING_MODEL $MODEL_NAME $BRANCH_NAME $QUESTION_PATH $SAVE_PATH $CACHE_DIR