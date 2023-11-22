import os
from kg_rag.utility import config_data


print("")
print("Starting to set up KG-RAG ...")
print("")

if os.path.exists(config_data["VECTOR_DB_PATH"]):
    print("vectorDB already exists!")
else:
    print("Creating vectorDB ...")
    from kg_rag.vectorDB.create_vectordb import create_vectordb
    create_vectordb()