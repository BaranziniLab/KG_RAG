#!/bin/bash

# Update PATH
echo 'export PATH="$HOME/conda/bin:$PATH"' >> $HOME/.bashrc
export PATH="$HOME/conda/bin:$PATH"

# Initialize conda
conda init bash

# Source the updated .bashrc to apply changes
source $HOME/.bashrc

# Create and activate the conda environment, and install requirements
conda create -y -n kg_rag python=3.10.9
source activate kg_rag
pip install -r /workspaces/KG_RAG/requirements.txt

# Ensure the conda environment is activated for future terminals
echo 'conda activate kg_rag' >> $HOME/.bashrc


# # Update PATH
# echo 'export PATH="$HOME/conda/bin:$PATH"' >> $HOME/.bashrc
# export PATH="$HOME/conda/bin:$PATH"

# # Initialize conda
# conda init

# # Create and activate the conda environment
# conda create -y -n kg_rag python=3.10.9
# echo 'conda activate kg_rag' >> $HOME/.bashrc
# pip install -r /workspaces/KG_RAG/requirements.txt
# source $HOME/.bashrc

# # conda create -y -n kg_rag python=3.10.9
# # echo 'conda activate kg_rag'
# # pip install -r /workspaces/KG_RAG/requirements.txt
