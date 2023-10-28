import sys
sys.path.insert(0, "../../")
from utility import *

CHAT_MODEL_ID = "gpt-4"
CHAT_DEPLOYMENT_ID = None
VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = "pritamdeka/S-PubMedBert-MS-MARCO"
QUESTION_PATH = "/data/somank/llm_data/analysis/one_hop_graph_traversal_questions.csv"
SAVE_PATH = "/data/somank/llm_data/analysis"



CONTEXT_VOLUME_LIST = [10, 50, 100, 150, 200]
# QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD_LIST = [10, 30, 60, 90]
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = 75
MAX_NUMBER_OF_CONTEXT_FOR_A_QUESTION = 150
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5


temperature = 0
if not CHAT_DEPLOYMENT_ID:
    CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID



system_prompt = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context and your inherent knowledge on the topic. Give your answer in the following JSON format:
    {{Compounds:<list of compounds>, Diseases:<list of diseases>}}
"""

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)    
    question_df = question_df.head(75)
    for context_index, context_volume in enumerate(CONTEXT_VOLUME_LIST):        
        answer_list = []
        for index, row in question_df.iterrows():
            question = row["text"]
            context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, context_volume, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)
            enriched_prompt = "Context: "+ context + "\n" + "Question: " + question
            output = get_GPT_response(enriched_prompt, system_prompt, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=temperature)
            if not output:
                time.sleep(5)
            answer_list.append((row["disease_1"], row["Compounds"], row["Diseases"], row["text"], output, context_volume))                           
        answer_df = pd.DataFrame(answer_list, columns=["disease", "compound_groundTruth", "disease_groundTruth", "text", "llm_answer", "context_volume"])
        save_name = "_".join(CHAT_MODEL_ID.split("-"))+"PubMedBert_node_retrieval_rag_based_drug_reporposing_questions_parameter_tuning_round_{}.csv".format(context_index+1)
        answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)
    print("Completed in {} min".format((time.time()-start_time)/60))
    
if __name__ == "__main__":
    main()
    
                
