from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.insert(0, "../../")
from utility import *


PROMPT_TYPE = sys.argv[1]

CHAT_MODEL_ID = "gpt-4"
CHAT_DEPLOYMENT_ID = None
VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_node_retrieval_rag_based_two_hop_mcq_from_monarch_response.csv"

# GPT config params
temperature = 0

if not CHAT_DEPLOYMENT_ID:
    CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

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

max_number_of_high_similarity_context_per_node = int(MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION/MAX_NODE_HITS)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

if PROMPT_TYPE == "mcq":
    system_prompt = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Based on that Context, provide your answer in the following JSON format for the Question asked.
    {{
      "answer": <correct answer>
    }}
    """
elif PROMPT_TYPE == "text":
    system_prompt = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context and your inherent knowledge on the topic. If you don't know the answer, report as "I don't know", don't try to make up an answer.
    """

embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)

def main():
    start_time = time.time()
    question = input("Enter your question : ")    
    print("Retrieving context from SPOKE graph...")
    context = retrieve_context(question)
    print("Context:\n")
    print(context)
    print("Here is my answer:")
    print("")
    context = "Context: "+ retrieve_context(question)
    enriched_prompt = context + "\n" + "Question: " + question
    output = get_GPT_response(enriched_prompt, system_prompt, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=temperature)
    print(output)

        
        
def retrieve_context(question):
    node_hits = vectorstore.similarity_search_with_score(question, k=MAX_NODE_HITS)
    print("Node hits:\n")    
    question_embedding = embedding_function.embed_query(question)
    node_context_extracted = ""
    for node in node_hits:
        node_name = node[0].page_content
        print(node_name)
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



