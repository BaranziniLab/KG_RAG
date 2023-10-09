from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import LlamaCpp
import sys

MODEL_PATH = sys.argv[1]
VECTOR_DB_PATH = sys.argv[2]
SENTENCE_EMBEDDING_MODEL = sys.argv[3]
N_GPU_LAYERS = sys.argv[4]
N_BATCH = sys.argv[5]
QUESTION_TYPE = sys.argv[6]

RETRIEVAL_SCORE_THRESH = 0.72



llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers = N_GPU_LAYERS,
            n_batch = N_BATCH,
            temperature=0,
            top_p=1,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), 
            verbose=True)




B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

system_prompt = """
You are a biomedical researcher. For answering the question at the end, you need to first read the Context provided and then answer the Question. If you don't know the answer, report as "I don't know", don't try to make up an answer.
"""
instruction = "Context:\n\n{context} \n\nQuestion: {question}"
template = get_prompt(instruction, system_prompt)












