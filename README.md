# Knowledge Graph-based Retrieval Augmented Generation (KG-RAG)

## Table of Contents
[What is KG-RAG](https://github.com/BaranziniLab/KG_RAG#what-is-kg-rag)

[Example use case of KG-RAG](https://github.com/BaranziniLab/KG_RAG#example-use-case-of-kg-rag)
 - [Prompting GPT without KG-RAG](https://github.com/BaranziniLab/KG_RAG#without-kg-rag)  
 - [Prompting GPT with KG-RAG](https://github.com/BaranziniLab/KG_RAG#with-kg-rag)

[How to run KG-RAG](https://github.com/BaranziniLab/KG_RAG#how-to-run-kg-rag)
 - [Step 1: Clone the repo](https://github.com/BaranziniLab/KG_RAG#step-1-clone-the-repo)
 - [Step 2: Create a virtual environment](https://github.com/BaranziniLab/KG_RAG#step-2-create-a-virtual-environment)
 - [Step 3: Install dependencies](https://github.com/BaranziniLab/KG_RAG#step-3-install-dependencies)
 - [Step 4: Update config.yaml](https://github.com/BaranziniLab/KG_RAG#step-4-update-configyaml)
 - [Step 5: Run the setup script](https://github.com/BaranziniLab/KG_RAG#step-5-run-the-setup-script)
 - [Step 6: Run KG-RAG from your terminal](https://github.com/BaranziniLab/KG_RAG#step-6-run-kg-rag-from-your-terminal)
    - [Using GPT](https://github.com/BaranziniLab/KG_RAG#using-gpt)
    - [Using GPT interactive mode](https://github.com/BaranziniLab/KG_RAG/blob/main/README.md#using-gpt-interactive-mode)
    - [Using Llama](https://github.com/BaranziniLab/KG_RAG#using-llama)
    - [Using Llama interactive mode](https://github.com/BaranziniLab/KG_RAG/blob/main/README.md#using-llama-interactive-mode)

## What is KG-RAG?

KG-RAG stands for Knowledge Graph-based Retrieval Augmented Generation.

### Start by watching the video of KG-RAG

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/a39d8f06-e16e-452d-af17-2e46b8dcdd62" controls="controls" style="max-width: 730px;">
</video>

It is a task agnostic framework that combines the explicit knowledge of a Knowledge Graph (KG) with the implicit knowledge of a Large Language Model (LLM). Here is the [arXiv preprint](https://arxiv.org/abs/2311.17330) of the work.

Here, we utilize a massive biomedical KG called [SPOKE](https://spoke.ucsf.edu/) as the provider for the biomedical context. SPOKE has incorporated over 40 biomedical knowledge repositories from diverse domains, each focusing on biomedical concept like genes, proteins, drugs, compounds, diseases, and their established connections. SPOKE consists of more than 27 million nodes of 21 different types and 53 million edges of 55 types [[Ref](https://doi.org/10.1093/bioinformatics/btad080)]


The main feature of KG-RAG is that it extracts "prompt-aware context" from SPOKE KG, which is defined as: 

**the minimal context sufficient enough to respond to the user prompt.** 

Hence, this framework empowers a general-purpose LLM by incorporating an optimized domain-specific 'prompt-aware context' from a biomedical KG.

## Example use case of KG-RAG
Following snippet shows the news from FDA [website](https://www.fda.gov/drugs/news-events-human-drugs/fda-approves-treatment-weight-management-patients-bardet-biedl-syndrome-aged-6-or-older) about the drug **"setmelanotide"** approved by FDA for weight management in patients with *Bardet-Biedl Syndrome*

<img src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/fc4d0b8d-0edb-461d-86c5-9d0d191bd97d" width="600" height="350">

### Ask GPT-3.5-Turbo about the above drug:

### WITHOUT KG-RAG

*Note: We are prompting GPT from the terminal, NOT from the chatGPT browser. Temperature parameter is set to 0 for all the analysis. Refer [this](https://github.com/BaranziniLab/KG_RAG/blob/main/config.yaml) yaml file for parameter setting*

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/9ca7cee1-5f53-4f2f-9b6b-eaeefbc78835" controls="controls" style="max-width: 730px;">
</video>

### WITH KG-RAG

*Note: Temperature parameter is set to 0 for all the analysis. Refer [this](https://github.com/BaranziniLab/KG_RAG/blob/main/config.yaml) yaml file for parameter setting*

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/13abe937-1ad9-43b3-8cdc-29fbc5fa525b" controls="controls" style="max-width: 730px;">
</video>

You can see that, KG-RAG was able to give the correct information about the FDA approved [drug](https://www.fda.gov/drugs/news-events-human-drugs/fda-approves-treatment-weight-management-patients-bardet-biedl-syndrome-aged-6-or-older).

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

### Step 5: Run the setup script
Note: Make sure you are in KG_RAG folder

Setup script runs in an interactive fashion.

Running the setup script will: 

- create disease vector database for KG-RAG
- download Llama model in your machine (optional, you can skip this and that is totally fine)

```
python -m kg_rag.run_setup
```

### Step 6: Run KG-RAG from your terminal
Note: Make sure you are in KG_RAG folder

You can run KG-RAG using GPT and Llama model. 

#### Using GPT

```
python -m kg_rag.rag_based_generation.GPT.text_generation <your favorite gpt model - "gpt-4" or "gpt-35-turbo">
```

Example:

Note: The following example was run on AWS p3.8xlarge EC2 instance.

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/13be98d6-92c7-4bb3-b455-d9d76c94e9b3" controls="controls" style="max-width: 730px;">
</video>

#### Using GPT interactive mode

This allows the user to go over each step of the process

```
python -m kg_rag.rag_based_generation.GPT.text_generation <your favorite gpt model - "gpt-4" or "gpt-35-turbo"> interactive
```

#### Using Llama
Note: If you haven't downloaded Llama during [setup](https://github.com/BaranziniLab/KG_RAG#step-5-run-the-setup-script) step, then when you run the following, it may take sometime since it will download the model first.

```
python -m kg_rag.rag_based_generation.Llama.text_generation
```

Example:

Note: The following example was run on AWS p3.8xlarge EC2 instance.

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/8b991622-2f99-4f91-856c-2d4a8a36578e" controls="controls" style="max-width: 730px;">
</video>

#### Using Llama interactive mode

This allows the user to go over each step of the process

```
python -m kg_rag.rag_based_generation.Llama.text_generation interactive
```















