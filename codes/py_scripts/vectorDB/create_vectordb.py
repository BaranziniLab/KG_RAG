import chromadb
import pickle
import time

start_time = time.time()

TEXTUAL_DATA_PATH_LIST = ["../data/disease_gene_textual_knowledge.pickle", "../data/disease_resembles_disease_textual_knowledge.pickle"]
db_name = "spoke_subset_knowledge"
metadata_name = "spoke_knowledge"

def get_data():
    textual_knowledge = []
    for TEXTUAL_DATA_PATH in TEXTUAL_DATA_PATH_LIST:
        with open(TEXTUAL_DATA_PATH, "rb") as f:
            textual_knowledge.extend(pickle.load(f))
    return textual_knowledge
        

# Persistent client
textual_knowledge = get_data()
client = chromadb.PersistentClient(path="../data/chroma_{}".format(db_name))
collection = client.create_collection(db_name)
collection.add(
    ids=[str(i) for i in range(0, len(textual_knowledge))], 
    documents=textual_knowledge,
    metadatas=[{"type": metadata_name} for _ in range(0, len(textual_knowledge))]
)

tot_time = round((time.time() - start_time)/(60*60), 2)
print("Database is populated in {} hr".format(tot_time))