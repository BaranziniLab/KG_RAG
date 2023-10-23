"""
This script shows how to tackle a 2-hop drug repurposing query using natural language:

Question:
    What drugs can be used to re-purpose psoriasis?

Implementation:
    Step 1: Find diseases that resemble to psoriasis.
    Step 2: Find compounds that treat those diseases
"""


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

MAX_NODE_HITS = 10
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = 0.95
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5

MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION = 150


"""
******************************************************************************************************
"""

max_number_of_high_similarity_context_per_node = int(MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION/MAX_NODE_HITS)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

system_prompt = "You are an expert biomedical researcher"

# Step 1
instruction_1 = """
What diseases resemble psoriasis?
Give the answer in the following format:
{{
answer : <list of diseases>
}}
"""

completion = openai.ChatCompletion.create(
    temperature=temperature, 
    top_p=1,
    deployment_id=CHAT_DEPLOYMENT_ID,
    model=CHAT_MODEL_ID,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction_1}
    ]
)

step1_answer = completion.choices[0].message["content"]
extracted_diseases = step1_answer.split("answer : [")[-1].split("]")[0]

# Step 2
instruction_2 = """
What Compounds are used to treat each of the following diseases?
{}
""".format(extracted_diseases)

completion = openai.ChatCompletion.create(
    temperature=temperature, 
    top_p=1,
    deployment_id=CHAT_DEPLOYMENT_ID,
    model=CHAT_MODEL_ID,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction_2}
    ]
)

step2_answer = completion.choices[0].message["content"]
print(step2_answer)



