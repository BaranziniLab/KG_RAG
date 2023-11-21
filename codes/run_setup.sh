echo ""
echo "Starting to set up KG-RAG ..."
echo ""

echo "Checking for disease concepts vectorDB, if it does not exist, it will be created ..."
CONFIG_FILE="../config.yaml"
VECTOR_DB_PATH=$(awk -F: '/VECTOR_DB_PATH/ {gsub(/ /, "", $2); print $2}' "$CONFIG_FILE")
VECTOR_DB_PATH=$(echo "$VECTOR_DB_PATH" | tr -d \')
if [ -d "$VECTOR_DB_PATH" ]; then
    echo "vectorDB already exists!"
else
    echo "Creating vectorDB ..."
    python -m py_scripts.vectorDB.create_vectordb
fi
