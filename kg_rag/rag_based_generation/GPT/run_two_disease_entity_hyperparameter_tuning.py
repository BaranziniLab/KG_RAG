'''
This script is used for hyperparameter tuning on two-hop graph traversal questions.
Hyperparameters are 'CONTEXT_VOLUME_LIST' and 'SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL_LIST'

This will run on two-hop graph traveral questions from the csv file and save the result as another csv file. 

Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import sys


CHAT_MODEL_ID = sys.argv[1]

CONTEXT_VOLUME_LIST = [10, 50, 100, 150, 200]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL_LIST = ["pritamdeka/S-PubMedBert-MS-MARCO", "sentence-transformers/all-MiniLM-L6-v2"]
SAVE_NAME_LIST = ["pubmedBert_based_two_hop_questions_parameter_tuning_round_{}.csv", "miniLM_based_two_hop_questions_parameter_tuning_round_{}.csv"]

QUESTION_PATH = config_data["TWO_DISEASE_ENTITY_FILE"]
SYSTEM_PROMPT = system_prompts["TWO_DISEASE_ENTITY_VALIDATION"]
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    for tranformer_index, sentence_embedding_model_for_context_retrieval in enumerate(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL_LIST):
        for context_index, context_volume in enumerate(CONTEXT_VOLUME_LIST):
            answer_list = []
            for index, row in question_df.iterrows():
                question = row["text"]
                context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, context_volume, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence)
                enriched_prompt = "Context: "+ context + "\n" + "Question: " + question
                output = get_GPT_response(enriched_prompt, system_prompt, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=temperature)
                if not output:
                    time.sleep(5)
                answer_list.append((row["disease_1"], row["disease_2"], row["central_nodes"], row["text"], output, context_volume))
        answer_df = pd.DataFrame(answer_list, columns=["disease_1", "disease_2", "central_nodes_groundTruth", "text", "llm_answer", "context_volume"])
        save_name = "_".join(CHAT_MODEL_ID.split("-"))+SAVE_NAME_LIST[tranformer_index].format(context_index+1)
        answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)
    print("Completed in {} min".format((time.time()-start_time)/60))
    
    
if __name__ == "__main__":
    main()
