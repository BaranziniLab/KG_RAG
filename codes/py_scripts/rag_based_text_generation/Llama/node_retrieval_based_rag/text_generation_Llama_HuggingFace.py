from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer
# from auto_gptq import exllama_set_max_input_length
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np



VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
BRANCH_NAME = "main"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"


# MAX_TOKEN_SIZE_OF_LLM = 4096
# CONTEXT_TOKEN_SIZE_FRACTION = 0.8

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

MAX_NODE_HITS = 3
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
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Importantly, if you don't see any relevant information in the Context with regard to the Question, answer "Don't know" as per the JSON format given below:

SYSTEM_PROMPT = """
You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context and your inherent knowledge on the topic. If you don't know the answer, report as "I don't know", don't try to make up an answer.
"""
INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)

vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)

def main():
    template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = get_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question = input("Enter your question : ")    
    print("Retrieving context from SPOKE graph...")
    context = retrieve_context(question)
    print("Here is my answer:")
    print("")
    output = llm_chain.run(context=context, question=question)

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def get_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                             cache_dir=CACHE_DIR
                                             )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,                                             
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                revision=BRANCH_NAME,
                                                cache_dir=CACHE_DIR
                                                )
    streamer = TextStreamer(tokenizer)

    pipe = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    max_new_tokens = 512,
                    do_sample = True,
                    top_k = 30,
                    num_return_sequences = 1,
                    streamer=streamer
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



if __name__ == "__main__":
    main()
