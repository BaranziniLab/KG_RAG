from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import pickle
import time


VECTOR_DB_PATH = "/data1/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data1/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUESTION_PATH = "/data1/somank/llm_data/analysis/test_questions.csv"
SAVE_PATH = "/data1/somank/llm_data/analysis"
SAVE_NAME = "extracted_context_of_true_false_test_questions.pickle"


LIST_OF_MAX_NODE_HITS = [1, 2, 3, 4, 5]
LIST_OF_MAX_NUMBER_OF_TOTAL_CONTEXT_FOR_A_QUESTION = [50, 100, 150]

QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = 95
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5


embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)

node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

def main():    
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)      
    question_dict_list = []
    for index, row in question_df.iterrows():
        question = row["text"]
        question_dict = {}        
        question_dict["question"] = question        
        context_dict_list = []        
        for eta in LIST_OF_MAX_NODE_HITS:
            for alpha_max in LIST_OF_MAX_NUMBER_OF_TOTAL_CONTEXT_FOR_A_QUESTION:
                context_dict = {}
                context_dict["param_combination"] = (eta, alpha_max)
                alpha_max_per_node = int(np.divide(alpha_max, eta))                
                context = retrieve_context(question, eta, alpha_max_per_node)
                context_dict["context"] = context
                context_dict_list.append(context_dict)
        question_dict["context"] = context_dict_list
        question_dict_list.append(question_dict)                        
    with open(os.path.join(SAVE_PATH, SAVE_NAME), "wb") as f:
        pickle.dump(question_dict_list, f)
    print("Completed in {} min".format((time.time()-start_time)/60))


def retrieve_context(question, max_node_hits, max_number_of_high_similarity_context_per_node):
    node_hits = vectorstore.similarity_search_with_score(question, k=max_node_hits)
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
