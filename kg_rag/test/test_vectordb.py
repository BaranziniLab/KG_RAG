from kg_rag.utility import *
import sys

VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]

print("Testing vectorDB loading ...")
print("")
try:
    vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
    print("vectorDB is loaded succesfully!")
except:
    print("vectorDB is not loaded. Check the path given in 'VECTOR_DB_PATH' of config.yaml")
    print("")
    sys.exit(1)
try:
    print("")
    print("Testing entity extraction ...")
    print("")
    entity = "psoriasis"
    print("Inputting '{}' as the entity to test ...".format(entity))    
    print("")
    node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
    extracted_entity = node_search_result[0][0].page_content
    print("Extracted entity is '{}'".format(extracted_entity))
    print("")
    if extracted_entity == "psoriasis":                
        print("Entity extraction is successful!")
        print("")
        print("vectorDB is correctly populated and is good to go!")
    else:
        print("Entity extraction is not successful. Make sure vectorDB is populated correctly. Refer 'How to run KG-RAG' Step 5")
        print("")
        sys.exit(1)
except:
    print("Entity extraction is not successful. Make sure vectorDB is populated correctly. Refer 'How to run KG-RAG' Step 5")
    print("")
    sys.exit(1)

    
    
    
