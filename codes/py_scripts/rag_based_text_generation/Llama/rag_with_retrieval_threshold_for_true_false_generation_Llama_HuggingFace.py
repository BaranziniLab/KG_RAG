from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
from auto_gptq import exllama_set_max_input_length
import torch
import pandas as pd
import os
import time
import sys


VECTOR_DB_PATH = sys.argv[1]
SENTENCE_EMBEDDING_MODEL = sys.argv[2]
MODEL_NAME = sys.argv[3]
BRANCH_NAME = sys.argv[4]
QUESTION_PATH = sys.argv[5]
SAVE_PATH = sys.argv[6]
CACHE_DIR = sys.argv[7]

"""
Check llm_notebooks/retrieval_optimization/retrieval_score_threshold_optimization_using_HF_tokenizer.ipynb
to see how optimal retrieval score threshold is obtained.
"""

RETRIEVAL_SCORE_THRESH = 0.55
MAX_TOKEN_SIZE_OF_LLM = 4096
QUESTION_TOKEN_SIZE = 120


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


SYSTEM_PROMPT = """
You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided and then provide your answer in the following JSON format:
{{
  "answer": "True"
}}
OR
{{
  "answer": "False"
}}
"""
INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"


embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)

vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, 
                     embedding_function=embedding_function)


def main():    
    llm = model(MODEL_NAME, BRANCH_NAME)               
    template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=BRANCH_NAME, use_auth_token=True, cache_dir=CACHE_DIR)
    template_tokens = tokenizer.tokenize(template)
    MAX_CONTEXT_TOKENS_IN_INPUT = MAX_TOKEN_SIZE_OF_LLM - len(template_tokens) - QUESTION_TOKEN_SIZE
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    start_time = time.time()
    SAVE_NAME = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_rag_based_binary_response.csv"
    question_df = pd.read_csv(QUESTION_PATH)        
    answer_list = []
    for index, row in question_df.iterrows():
        question = row["text"]
        context = retrieve_context(question)
        context_tokens = tokenizer.tokenize(context)
        if len(context_tokens) > MAX_CONTEXT_TOKENS_IN_INPUT:
            context = ''.join([word if word != '<0x0A>' else '\n' for word in context_tokens[0:MAX_CONTEXT_TOKENS_IN_INPUT]]).replace('▁',' ').strip()
        output = llm_chain.run(context=context, question=question)
        answer_list.append((row["text"], row["label"], output))
    answer_df = pd.DataFrame(answer_list, columns=["question", "label", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, SAVE_NAME), index=False, header=True)    
    print("Completed in {} min".format((time.time()-start_time)/60))
    

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def model(MODEL_NAME, BRANCH_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                             use_auth_token=True,
                                             revision=BRANCH_NAME,
                                             cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,                                             
                                        device_map='auto',
                                        torch_dtype=torch.float16,
                                        use_auth_token=True,
                                        revision=BRANCH_NAME,
                                        cache_dir=CACHE_DIR
                                        )
#     model = exllama_set_max_input_length(model, MAX_TOKEN_SIZE_OF_LLM)
    # gptq_config = GPTQConfig(bits=4, group_size=64, desc_act=True)
    pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                max_new_tokens = 512,
                do_sample = False
                )
    
    llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":0, "top_p":1})
    return llm

def retrieve_context(question):
    search_result = vectorstore.similarity_search_with_score(question, k=10000)
    score_range = (search_result[-1][-1] - search_result[0][-1]) / (search_result[-1][-1] + search_result[0][-1])
    thresh = RETRIEVAL_SCORE_THRESH*score_range
    retrieved_context = ""
    for item in search_result:
        item_score = (search_result[-1][-1] - item[-1]) / (search_result[-1][-1] + item[-1])
        if item_score < thresh:
            break
        retrieved_context += item[0].page_content
        retrieved_context += "\n"
    return retrieved_context



        
        
if __name__ == "__main__":
    main()


