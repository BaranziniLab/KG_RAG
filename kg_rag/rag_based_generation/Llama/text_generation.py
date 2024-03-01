from langchain import PromptTemplate, LLMChain
from kg_rag.utility import *
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-i', type=bool, default=False, help='Flag for interactive mode')
parser.add_argument('-m', type=str, default='method-1', help='Method to choose for Llama model')
parser.add_argument('-e', type=bool, default=False, help='Flag for showing evidence of association from the graph')
args = parser.parse_args()

INTERACTIVE = args.i
METHOD = args.m
EDGE_EVIDENCE = bool(args.e)


SYSTEM_PROMPT = system_prompts["KG_RAG_BASED_TEXT_GENERATION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
MODEL_NAME = config_data["LLAMA_MODEL_NAME"]
BRANCH_NAME = config_data["LLAMA_MODEL_BRANCH"]
CACHE_DIR = config_data["LLM_CACHE_DIR"]


INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

def main():
    print(" ")
    question = input("Enter your question : ")    
    if not INTERACTIVE:
        template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = llama_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR, stream=True, method=METHOD) 
        llm_chain = LLMChain(prompt=prompt, llm=llm)            
        print("Retrieving context from SPOKE graph...")
        context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, EDGE_EVIDENCE)
        print("Here is the KG-RAG based answer using Llama:")
        print("")
        output = llm_chain.run(context=context, question=question)
    else:
        interactive(question, vectorstore, node_context_df, embedding_function_for_context_retrieval, "llama", EDGE_EVIDENCE, SYSTEM_PROMPT, llama_method=METHOD)



        



if __name__ == "__main__":
    main()
