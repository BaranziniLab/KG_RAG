from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
from auto_gptq import exllama_set_max_input_length
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import time
import sys
sys.path.insert(0, "../../../")
from utility import *


VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = "pritamdeka/S-PubMedBert-MS-MARCO"
QUESTION_PATH = "/data/somank/llm_data/analysis/drug_repurposing_questions.csv"
SAVE_PATH = "/data/somank/llm_data/analysis"
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
BRANCH_NAME = "main"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"

CONTEXT_VOLUME = 150
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = 75
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5


save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_entity_recognition_based_node_retrieval_rag_based_drug_repurposing_questions_response.csv"


torch.cuda.empty_cache()


node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context and your inherent knowledge on the topic. Give your answer in the following JSON format:
    {{Compounds:<list of compounds>, Diseases:<list of diseases>}}
"""

INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"


embedding_function_for_node_retrieval = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function_for_node_retrieval)


def main():
    start_time = time.time()
    llm = model(MODEL_NAME, BRANCH_NAME)               
    template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)    
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    for index, row in question_df.iterrows():
        question = row["text"]
        context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)
        output = llm_chain.run(context=context, question=question)
        answer_list.append((row["disease_1"], row["Compounds"], row["Diseases"], row["text"], output))
    answer_df = pd.DataFrame(answer_list, columns=["disease", "compound_groundTruth", "disease_groundTruth", "text", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)
    print("Completed in {} min".format((time.time()-start_time)/60))

def get_prompt(instruction, new_system_prompt):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def model(MODEL_NAME, BRANCH_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                             revision=BRANCH_NAME,
                                             cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,                                             
                                        device_map='auto',
                                        torch_dtype=torch.float16,
                                        revision=BRANCH_NAME,
                                        cache_dir=CACHE_DIR
                                        )
    pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                max_new_tokens = 512,
                do_sample = True
                )    
    llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":0, "top_p":1})
    return llm






if __name__ == "__main__":
    main()

        

