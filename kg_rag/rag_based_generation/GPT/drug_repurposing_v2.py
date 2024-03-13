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

SYSTEM_PROMPT = system_prompts["DRUG_REPURPOSING_V2"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

print('')
question = input("Question : ")

question_template = f'''
To the question asked at the end, answer by referring the context. 
See example below
Example 1:
    Question:
    What drugs can be repurposed for disease X?
    Context:
    Compound Alizapride DOWNREGULATES Gene APOE and Provenance of this association is XX. Gene APOE ASSOCIATES Disease X  and Provenance of this association is YY. Gene TTR encodes Protein Transthyretin (ATTR)  and Provenance of this association is ZZ. Compound Acetylcysteine treats Disease X  and Provenance of this association is PP.
    Answer:
    Since Alizapride downregulates gene APOE (Provenance XX) and APOE is associated with Disease X (Provenance YY), Alizapride can be repurposed to treat Disease X. p-value for these associations is XXXX and z-score values for these associations is YYYY.
Question:
{question} 
'''

def main():
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


