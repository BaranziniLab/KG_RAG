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
QUESTION_PATH = "/data/somank/llm_data/analysis/disease_two_hop_validation_data.csv"
SAVE_PATH = "/data/somank/llm_data/analysis"

MAX_NODE_HITS_LIST = [1, 10, 20, 30]
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD_LIST = [10, 30, 60, 90]
MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION = 150
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5

temperature = 0
if not CHAT_DEPLOYMENT_ID:
    CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID


node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

system_prompt = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context and your inherent knowledge on the topic. Give your answer in the following JSON format:
    {{Nodes:<list of nodes>}}
"""

embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)
