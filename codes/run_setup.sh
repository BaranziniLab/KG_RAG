echo "Starting setting up KG-RAG ..."

echo "Checking vectorDB for disease concepts ..."
VECTOR_DB_PATH=$(cat "$CONFIG_FILE" | grep "VECTOR_DB_PATH" | cut -d ':' -f2 | tr -d '[:space:]')
if [ -d "$VECTOR_DB_PATH" ]; then
    echo "vectorDB already exists!"
else
    echo "Creating vectorDB ..."
    python py_scripts/vectorDB/create_vectordb.py
fi