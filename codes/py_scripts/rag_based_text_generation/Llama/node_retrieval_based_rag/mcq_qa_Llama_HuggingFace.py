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


VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
BRANCH_NAME = "main"
QUESTION_PATH = "/data/somank/llm_data/analysis/test_questions_two_hop_mcq_from_monarch.csv"
SAVE_PATH = "/data/somank/llm_data/analysis"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"

SAVE_NAME = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_node_retrieval_rag_based_two_hop_mcq_from_monarch_response.csv"
"""
****************************************************************************************************** 
                        Retrieval parameters
Following parameter decides how many maximum associations to consider from the knowledge graph to answer a question.

If a node hit for a question has N degree, then we will consider a maximum of 
MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION/MAX_NODE_HITS 
associations out of that N.

In other words, an upper cap of "MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION" associations will be considered in total across all node hits to answer a question. 

Hence, MAX_NODE_HITS and MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION can be considered as the hyperparameters that control the information flow from knowledge graph to LLM. They can be tweaked based on the complexity of the question dataset that needs to be answered.

It also controls the token size that goes as input to the LLM.
"""

MAX_NODE_HITS = 30
MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION = 150
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = 95
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5

"""
******************************************************************************************************
"""

torch.cuda.empty_cache()

max_number_of_high_similarity_context_per_node = int(MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION/MAX_NODE_HITS)
# context_token_size = int(CONTEXT_TOKEN_SIZE_FRACTION*MAX_TOKEN_SIZE_OF_LLM)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = """
You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Based on that Context, provide your answer in the following JSON format for the Question asked:
{{
  "answer": <correct answer>
}}
"""

INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)

vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)

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
#     model = exllama_set_max_input_length(model, MAX_TOKEN_SIZE_OF_LLM)
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


def retrieve_context(question):
    node_hits = vectorstore.similarity_search_with_score(question, k=MAX_NODE_HITS)
    question_embedding = embedding_function.embed_query(question)
    node_context_extracted = ""
    for node in node_hits:
        node_name = node[0].page_content
        node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        node_context_list = node_context.split(". ")        
        node_context_embeddings = embedding_function.embed_documents(node_context_list)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities], QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD)
        high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY]
        if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
            high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
        node_context_extracted += ". ".join(high_similarity_context)
        node_context_extracted += ". "
    return node_context_extracted

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
        context = retrieve_context(question)
        output = llm_chain.run(context=context, question=question)
        answer_list.append((row["text"], row["correct_node"], output))
    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, SAVE_NAME), index=False, header=True) 
    print("Completed in {} min".format((time.time()-start_time)/60))

# question = input("Enter your question : ")
# llm = model(MODEL_NAME, BRANCH_NAME)               
# template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
# prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# context = retrieve_context(question)
# output = llm_chain.run(context=context, question=question)
# print(output)


if __name__ == "__main__":
    main()
    
    
