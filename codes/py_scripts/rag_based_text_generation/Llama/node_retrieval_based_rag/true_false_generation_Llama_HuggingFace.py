from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
from auto_gptq import exllama_set_max_input_length
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import time
import sys


VECTOR_DB_PATH = "/data1/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data1/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
BRANCH_NAME = "main"
QUESTION_PATH = "/data1/somank/llm_data/analysis/test_questions.csv"
SAVE_PATH = "/data1/somank/llm_data/analysis"
CACHE_DIR = "/data1/somank/llm_data/llm_models/huggingface"


MAX_TOKEN_SIZE_OF_LLM = 4096
# CONTEXT_TOKEN_SIZE_FRACTION = 0.8
MAX_NODE_HITS = 5
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = 95
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5

# context_token_size = int(CONTEXT_TOKEN_SIZE_FRACTION*MAX_TOKEN_SIZE_OF_LLM)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

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
OR
{{
  "answer": "Don't know"
}}
"""
INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)

vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)


def main():    
    start_time = time.time()
    llm = model(MODEL_NAME, BRANCH_NAME)               
    template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    SAVE_NAME = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_node_retrieval_rag_based_response.csv"
    question_df = pd.read_csv(QUESTION_PATH)  
    answer_list = []
    for index, row in question_df.iterrows():
        question = row["text"]
        context = retrieve_context(question)
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
    pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                max_new_tokens = 512,
                do_sample = True,
                top_k = 30,
                num_return_sequences = 1
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
        percentile_threshold = np.percentile(similarities, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD)
        high_similarity_indices = [index for index, similarity in enumerate(similarities) if similarity > percentile_threshold and similarity > QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY]
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
        node_context_extracted += ". ".join(high_similarity_context)
        node_context_extracted += ". "

        

    
    
if __name__ == "__main__":
    main()
    
    