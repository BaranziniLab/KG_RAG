MODEL_PATH="/data1/somank/llm_data/llm_models/llama-2-13b-chat.Q5_K_M.gguf"
N_GPU_LAYERS=40
N_BATCH=1024


python ../py_scripts/llm_using_llama_cpp_and_langchain.py $MODEL_PATH $N_GPU_LAYERS $N_BATCH