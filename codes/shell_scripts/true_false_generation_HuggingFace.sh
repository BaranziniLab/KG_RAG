MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
BRANCH_NAME="main"
QUESTION_PATH="/data/somank/llm_data/analysis/test_questions.csv"
SAVE_PATH="/data/somank/llm_data/analysis"
STREAM="False"
CACHE_DIR="/data/somank/llm_data/llm_models/huggingface"

python ../py_scripts/prompt_based_text_generation/true_false_generation_HuggingFace.py $MODEL_NAME $BRANCH_NAME $QUESTION_PATH $SAVE_PATH $STREAM $CACHE_DIR