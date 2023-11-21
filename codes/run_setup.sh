echo "Starting to set up KG-RAG ..."
echo ""

echo "Checking vectorDB for disease concepts ..."
CONFIG_FILE="../config.yaml"
VECTOR_DB_PATH=$(cat "$CONFIG_FILE" | grep "VECTOR_DB_PATH" | cut -d ':' -f2 | tr -d '[:space:]')
if [ -d "$VECTOR_DB_PATH" ]; then
    echo "vectorDB already exists!"
else
    echo "Creating vectorDB ..."
    python -m py_scripts.vectorDB.create_vectordb
fi