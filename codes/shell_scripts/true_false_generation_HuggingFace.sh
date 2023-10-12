MODEL_NAME="TheBloke/Llama-2-13b-Chat-GPTQ"
BRANCH_NAME="gptq-4bit-64g-actorder_True"
QUESTION_PATH="/data1/somank/llm_data/analysis/test_questions.csv"
SAVE_PATH="/data1/somank/llm_data/analysis"
STREAM="False"
CACHE_DIR="/data1/somank/llm_data/llm_models/huggingface"

python ../py_scripts/prompt_based_text_generation/true_false_generation_HuggingFace.py $MODEL_NAME $BRANCH_NAME $QUESTION_PATH $SAVE_PATH $STREAM $CACHE_DIR