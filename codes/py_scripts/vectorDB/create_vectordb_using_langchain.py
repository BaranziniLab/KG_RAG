from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pickle
import time
import sys


DATA_PATH = sys.argv[1]
CHUNK_SIZE = int(sys.argv[2])
CHUNK_OVERLAP = int(sys.argv[3])
BATCH_SIZE = int(sys.argv[4])
SENTENCE_EMBEDDING_MODEL = sys.argv[5]
VECTOR_DB_NAME = sys.argv[6]


def load_data():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    metadata_list = list(map(lambda x:{"source": x + " from SPOKE knowledge graph"}, data))
#     metadata_list = list(map(lambda x:{"node information":x.split("(")[0].split("Following is the contextual information about the ")[-1] + "from SPOKE knowledge graph"}, data))
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
    end_time = round((time.time() - start_time)/(60*60), 2)
    print("VectorDB is created in {} hrs".format(end_time))


if __name__ == "__main__":
    create_vectordb()
    
