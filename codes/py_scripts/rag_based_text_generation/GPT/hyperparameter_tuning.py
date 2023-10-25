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

CHAT_MODEL_ID = "gpt-4"
CHAT_DEPLOYMENT_ID = None
VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUESTION_PATH = "/data/somank/llm_data/analysis/drug_reporposing_questions.csv"
SAVE_PATH = "/data/somank/llm_data/analysis"



MAX_NODE_HITS_LIST = [1, 10, 20, 30]
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD_LIST = [10, 30, 50, 70, 90]
MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION = 150
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5



temperature = 0
if not CHAT_DEPLOYMENT_ID:
    CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID


node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

system_prompt = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context and your inherent knowledge on the topic. Give your answer in the following JSON format:
    {{Compounds:<list of compounds>, Diseases:<list of diseases>}}
"""

embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)    
    for node_hit_index, MAX_NODE_HITS in enumerate(MAX_NODE_HITS_LIST):
        answer_list = []
        for QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD in QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD_LIST:     
            max_number_of_high_similarity_context_per_node = int(MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION/MAX_NODE_HITS)
            for index, row in question_df.iterrows():
                question = row["text"]
                context = "Context: "+ retrieve_context(question, MAX_NODE_HITS, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, max_number_of_high_similarity_context_per_node)
                enriched_prompt = context + "\n" + "Question: " + question
                output = get_GPT_response(enriched_prompt, system_prompt, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=temperature)
                answer_list.append((row["disease_1"], row["Compounds"], row["Diseases"], row["text"], output, MAX_NODE_HITS, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD))                
        answer_df = pd.DataFrame(answer_list, columns=["disease", "compound_groundTruth", "disease_groundTruth", "text", "llm_answer", "max_node_hits", "context_similarity_threshold"])
        save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_node_retrieval_rag_based_drug_reporposing_questions_parameter_tuning_round_{}.csv".format(node_hit_index+1)
        answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)
        time.sleep(10)
    print("Completed in {} min".format((time.time()-start_time)/60))
    

def retrieve_context(question, MAX_NODE_HITS, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, max_number_of_high_similarity_context_per_node):
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
    
                
