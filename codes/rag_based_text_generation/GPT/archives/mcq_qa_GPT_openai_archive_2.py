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



CHAT_MODEL_ID = "gpt-35-turbo"
CHAT_DEPLOYMENT_ID = None
VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUESTION_PATH = "/data/somank/llm_data/analysis/test_questions_two_hop_mcq_from_monarch_and_robokop.csv"
SAVE_PATH = "/data/somank/llm_data/analysis"

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_entity_recognition_based_node_retrieval_rag_based_two_hop_mcq_from_monarch_and_robokop_response.csv"

# GPT config params
temperature = 0

if not CHAT_DEPLOYMENT_ID:
    CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID


CONTEXT_VOLUME = 150
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = 75
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5


node_context_df = pd.read_csv(NODE_CONTEXT_PATH)


    
system_prompt = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Based on that Context, provide your answer in the following JSON format for the Question asked.
    {{
      "answer": <correct answer>
    }}
"""

embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    for index, row in question_df.iterrows():
        print(index)
        question = "Question: "+ row["text"]
        context = "Context: "+ retrieve_context(row["text"], vectorstore, embedding_function, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)
        enriched_prompt = context + "\n" + question
        output = get_GPT_response(enriched_prompt, system_prompt, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=temperature)
        answer_list.append((row["text"], row["correct_node"], output))
    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True) 
    print("Completed in {} min".format((time.time()-start_time)/60))

        
        
if __name__ == "__main__":
    main()


