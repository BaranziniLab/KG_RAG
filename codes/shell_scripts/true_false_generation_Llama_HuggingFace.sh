MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
BRANCH_NAME="main"
QUESTION_PATH="/data1/somank/llm_data/analysis/test_questions_two_hop.csv"
SAVE_PATH="/data1/somank/llm_data/analysis"
CACHE_DIR="/data1/somank/llm_data/llm_models/huggingface"

python ../py_scripts/prompt_based_text_generation/Llama/true_false_generation_Llama_HuggingFace.py $MODEL_NAME $BRANCH_NAME $QUESTION_PATH $SAVE_PATH $CACHE_DIR