import sys
sys.path.insert(0,"../")
from utility import *


VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = "sentence-transformers/all-MiniLM-L6-v2"
QUESTION_PATH = "/data/somank/llm_data/analysis/test_questions_one_hop_true_false.csv"
SAVE_PATH = "/data/somank/llm_data/analysis"

SAVE_NAME = "true_false_question_spoke_map.csv"

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    result = []
    for index, row in question_df.iterrows():
        question = row["text"]
        entities = disease_entity_extractor(question)
        node_hits = []
        score = []
        for entity in entities:
            node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
            node_hits.append(node_search_result[0][0].page_content)
            score.append(node_search_result[-1])
        result.append((row["text"], row["label"], node_hits, score))
    mapped_question_df = pd.DataFrame(result, columns=["text", "label", "node_hits", "score"])
    mapped_question_df.to_csv(os.path.join(SAVE_PATH, SAVE_NAME), index=False, header=True)
    print("Completed in {} min".format((time.time()-start_time)/60))
    
if __name__ == "__main__":
    main()

    
