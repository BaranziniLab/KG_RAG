from langchain import PromptTemplate, LLMChain
import sys
sys.path.insert(0, "../../../")
from utility import *


VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = "pritamdeka/S-PubMedBert-MS-MARCO"
QUESTION_PATH = "/data/somank/llm_data/analysis/drug_repurposing_questions.csv"
SAVE_PATH = "/data/somank/llm_data/analysis"
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
BRANCH_NAME = "main"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"


CONTEXT_VOLUME = 150
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = 75
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5


save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_entity_recognition_based_node_retrieval_rag_based_drug_repurposing_questions_response.csv"



SYSTEM_PROMPT = """
    You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context and your inherent knowledge on the topic. Give your answer in the following JSON format:
    {{Compounds:<list of compounds>, Diseases:<list of diseases>}}
"""

INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)



def main():
    start_time = time.time()
    llm = llama_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR, max_new_tokens=1024)               
    template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)    
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    for index, row in question_df.iterrows():
        question = row["text"]
        context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)
        output = llm_chain.run(context=context, question=question)
        answer_list.append((row["disease_1"], row["Compounds"], row["Diseases"], row["text"], output))
    answer_df = pd.DataFrame(answer_list, columns=["disease", "compound_groundTruth", "disease_groundTruth", "text", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)
    print("Completed in {} min".format((time.time()-start_time)/60))







if __name__ == "__main__":
    main()

        

