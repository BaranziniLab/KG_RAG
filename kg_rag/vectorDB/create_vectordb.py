import pickle
from kg_rag.utility import RecursiveCharacterTextSplitter, Chroma, SentenceTransformerEmbeddings, config_data, time


DATA_PATH = config_data["VECTOR_DB_DISEASE_ENTITY_PATH"]
VECTOR_DB_NAME = config_data["VECTOR_DB_PATH"]
CHUNK_SIZE = int(config_data["VECTOR_DB_CHUNK_SIZE"])
CHUNK_OVERLAP = int(config_data["VECTOR_DB_CHUNK_OVERLAP"])
BATCH_SIZE = int(config_data["VECTOR_DB_BATCH_SIZE"])
SENTENCE_EMBEDDING_MODEL = config_data["VECTOR_DB_SENTENCE_EMBEDDING_MODEL"]


def load_data():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    metadata_list = list(map(lambda x:{"source": x + " from SPOKE knowledge graph"}, data))
    return data, metadata_list

def create_vectordb():
    start_time = time.time()
    data, metadata_list = load_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.create_documents(data, metadatas=metadata_list)
    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    vectorstore = Chroma(embedding_function=SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL), 
                         persist_directory=VECTOR_DB_NAME)
    for batch in batches:
        vectorstore.add_documents(documents=batch)
    end_time = round((time.time() - start_time)/(60), 2)
    print("VectorDB is created in {} mins".format(end_time))


if __name__ == "__main__":
    create_vectordb()
    
