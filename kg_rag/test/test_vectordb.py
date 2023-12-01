from kg_rag.utility import *
import sys

VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]

print("Testing vectorDB loading ...")
try:
    vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
    print("vectorDB is loaded succesfully!")
    print("Testing entity extraction ...")
    entity = "psoriasis"
    print("Inputting '{}' as the entity".format(entity))    
    node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
    extracted_entity = node_search_result[0][0].page_content
    print("Extracted entity is ", extracted_entity)
    if extracted_entity == "psoriasis":                
        print("Entity extraction is successful!")
    else:
        print("Entity extraction is not successful. Make sure vectorDB is populated correctly. Refer 'How to run KG-RAG' Step 5")
        sys.exit(1)
except:
    print("vectorDB is not loaded. Check the path given in 'VECTOR_DB_PATH' of config.yaml")
    sys.exit(1)
    
    
    
