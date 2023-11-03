import sys
sys.path.insert(0, "../../")
from utility import *


PROMPT_TYPE = sys.argv[1]
CONTEXT_VOLUME = int(sys.argv[2])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(sys.argv[3])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(sys.argv[4])

CHAT_MODEL_ID = "gpt-4"
CHAT_DEPLOYMENT_ID = None
VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = "pritamdeka/S-PubMedBert-MS-MARCO"


# GPT config params
temperature = 0

if not CHAT_DEPLOYMENT_ID:
    CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID



if PROMPT_TYPE == "mcq":
    system_prompt = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Based on that Context, provide your answer in the following JSON format for the Question asked.
    {{
      "answer": <correct answer>
    }}
    """
elif PROMPT_TYPE == "text":
    system_prompt = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context.
    """

    
vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

def main():
    start_time = time.time()
    question = input("Enter your question : ")    
    print("Retrieving context from SPOKE graph...")
    context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)
    print("Context:\n")
    print(context)
    print("Here is my answer:")
    print("")
    enriched_prompt = "Context: "+ context + "\n" + "Question: " + question
    output = get_GPT_response(enriched_prompt, system_prompt, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=temperature)
    print(output)

    
            
#     You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context. Give your answer in the following JSON format:
#     {{Nodes:<list of nodes>}}
#     If you don't know the answer report as an empty list as follows:
#         {{Nodes:[]}}
    
#     You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context and your inherent knowledge on the topic. If you don't know the answer, report as "I don't know", don't try to make up an answer.


if __name__ == "__main__":
    main()



