'''
This script takes a question from the user in an interactive fashion and returns the KG-RAG based response in real time
Before running this script, make sure to configure config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-g', type=str, default='gpt-35-turbo', help='GPT model selection')
parser.add_argument('-i', type=bool, default=False, help='Flag for interactive mode')
parser.add_argument('-e', type=bool, default=False, help='Flag for showing evidence of association from the graph')
args = parser.parse_args()

CHAT_MODEL_ID = args.g
INTERACTIVE = args.i
EDGE_EVIDENCE = bool(args.e)


SYSTEM_PROMPT = system_prompts["KG_RAG_BASED_TEXT_GENERATION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]

# can not specify CHAT_DEPLOYMENT_ID if GPT_API_TYPE is 'openai'.Otherwise it will be not work
CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID if openai.api_type == "azure" else None


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

def main():
    print(" ")
    question = input("Enter your question : ")
    if not INTERACTIVE:
        print("Retrieving context from SPOKE graph...")
        context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, EDGE_EVIDENCE)
        print("Here is the KG-RAG based answer:")
        print("")
        enriched_prompt = "Context: "+ context + "\n" + "Question: " + question
        output = get_GPT_response(enriched_prompt, SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=TEMPERATURE)
        stream_out(output)
    else:
        interactive(question, vectorstore, node_context_df, embedding_function_for_context_retrieval, CHAT_MODEL_ID, EDGE_EVIDENCE, SYSTEM_PROMPT)

                

if __name__ == "__main__":
    main()



