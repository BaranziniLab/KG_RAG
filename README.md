# Knowledge Graph-based Retrieval Augmented Generation (KG-RAG)

## Table of Contents
[What is KG-RAG](https://github.com/BaranziniLab/KG_RAG#what-is-kg-rag)

[Example use case of KG-RAG](https://github.com/BaranziniLab/KG_RAG#example-use-case-of-kg-rag)

 - [Prompting GPT without KG-RAG](https://github.com/BaranziniLab/KG_RAG#without-kg-rag)
  
 - [Prompting GPT with KG-RAG](https://github.com/BaranziniLab/KG_RAG#with-kg-rag)

## What is KG-RAG?

KG-RAG is a task agnostic framework that combines the explicit knwoledge of a Knowledge Graph (KG) with the implicit knwoledge of a Large Language Model (LLM). 

The main feature of KG-RAG is that it extracts "prompt-aware context" from the KG, which is defined as: 

**the minimal context sufficient enough to respond to the user prompt.** 

Hence, the framework provides context by optimizing the input token space of the LLM.

## Example use case of KG-RAG
Following snippet shows the news from FDA website about the drug **"setmelanotide"** approved by FDA for weight management in patients with *Bardet-Biedl Syndrome*

<img src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/fc4d0b8d-0edb-461d-86c5-9d0d191bd97d" width="600" height="350">

### Ask GPT-3.5-Turbo about the above drug:

### WITHOUT KG-RAG

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/9ca7cee1-5f53-4f2f-9b6b-eaeefbc78835" controls="controls" style="max-width: 730px;">
</video>

### WITH KG-RAG

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/77ec19b6-e84d-4cbb-9d6d-8305e6f31b71" controls="controls" style="max-width: 730px;">
</video>

## How to run KG-RAG

### Step 1: Clone the repo

Clone this repository. All Biomedical data used in the paper are uploaded to this repository, hence you don't have to download that separately.

### Step 2: Create a virtual environment
Note: Scripts in this repository were run using python 3.10.9
```
conda create -n kg_rag python=3.10.9
conda activate kg_rag
cd KG_RAG
```

### Step 3: Install dependencies

```
pip install -r requirements.txt
```

### Step 4: Update config.yaml 

[config.yaml](https://github.com/BaranziniLab/KG_RAG/blob/main/config.yaml) holds all the necessary information required to run the scripts in your machine. Make sure to populate [this](https://github.com/BaranziniLab/KG_RAG/blob/main/config.yaml) yaml file accordingly.

Note: There is another yaml file called [system_prompts.yaml](https://github.com/BaranziniLab/KG_RAG/blob/main/system_prompts.yaml). This is already populated and it holds all the system prompts used in the KG-RAG framework.

### Step 5: Run setup
Note: Make sure you are in KG_RAG folder

Setup script runs in an interactive fashion. 

Running the setup script will: 

- create vector database for the disease concepts
- download Llama model in your machine (optional, you can skip this and that is totally fine)

```
python -m kg_rag.run_setup
```





