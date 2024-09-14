<p align="center">
  <img src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/0b2f5b42-761e-4d5b-8d6f-77c8b965f017" width="450">
</p>




## Table of Contents
[What is KG-RAG](https://github.com/BaranziniLab/KG_RAG#what-is-kg-rag)

[Example use case of KG-RAG](https://github.com/BaranziniLab/KG_RAG#example-use-case-of-kg-rag)
 - [Prompting GPT without KG-RAG](https://github.com/BaranziniLab/KG_RAG#without-kg-rag)  
 - [Prompting GPT with KG-RAG](https://github.com/BaranziniLab/KG_RAG#with-kg-rag)
 - [Example notebook for KG-RAG with GPT](https://github.com/BaranziniLab/KG_RAG/blob/main/notebooks/kg_rag_based_gpt_prompts.ipynb)

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
  - [Command line arguments for KG-RAG](https://github.com/BaranziniLab/KG_RAG?tab=readme-ov-file#command-line-arguments-for-kg-rag)
  
[BiomixQA: Benchmark dataset](https://github.com/BaranziniLab/KG_RAG/tree/main?tab=readme-ov-file#biomixqa-benchmark-dataset)

[Citation](https://github.com/BaranziniLab/KG_RAG/blob/main/README.md#citation)


## What is KG-RAG?

KG-RAG stands for Knowledge Graph-based Retrieval Augmented Generation.

### Start by watching the video of KG-RAG

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/86e5b8a3-eb58-4648-95a4-271e9c69b4ed" controls="controls" style="max-width: 730px;">
</video>

It is a task agnostic framework that combines the explicit knowledge of a Knowledge Graph (KG) with the implicit knowledge of a Large Language Model (LLM). Here is the [arXiv preprint](https://arxiv.org/abs/2311.17330) of the work.

Here, we utilize a massive biomedical KG called [SPOKE](https://spoke.ucsf.edu/) as the provider for the biomedical context. SPOKE has incorporated over 40 biomedical knowledge repositories from diverse domains, each focusing on biomedical concept like genes, proteins, drugs, compounds, diseases, and their established connections. SPOKE consists of more than 27 million nodes of 21 different types and 53 million edges of 55 types [[Ref](https://doi.org/10.1093/bioinformatics/btad080)]


The main feature of KG-RAG is that it extracts "prompt-aware context" from SPOKE KG, which is defined as: 

**the minimal context sufficient enough to respond to the user prompt.** 

Hence, this framework empowers a general-purpose LLM by incorporating an optimized domain-specific 'prompt-aware context' from a biomedical KG.

## Example use case of KG-RAG
Following snippet shows the news from FDA [website](https://www.fda.gov/drugs/news-events-human-drugs/fda-approves-treatment-weight-management-patients-bardet-biedl-syndrome-aged-6-or-older) about the drug **"setmelanotide"** approved by FDA for weight management in patients with *Bardet-Biedl Syndrome*

<img src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/fc4d0b8d-0edb-461d-86c5-9d0d191bd97d" width="600" height="350">

### Ask GPT-4 about the above drug:

### WITHOUT KG-RAG

*Note: This example was run using KG-RAG v0.3.0. We are prompting GPT from the terminal, NOT from the chatGPT browser. Temperature parameter is set to 0 for all the analysis. Refer [this](https://github.com/BaranziniLab/KG_RAG/blob/main/config.yaml) yaml file for parameter setting*

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/dbabb812-2a8a-48b6-9785-55b983cb61a4" controls="controls" style="max-width: 730px;">
</video>

### WITH KG-RAG

*Note: This example was run using KG-RAG v0.3.0. Temperature parameter is set to 0 for all the analysis. Refer [this](https://github.com/BaranziniLab/KG_RAG/blob/main/config.yaml) yaml file for parameter setting*

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/acd08954-a496-4a61-a3b1-8fc4e647b2aa" controls="controls" style="max-width: 730px;">
</video>

You can see that, KG-RAG was able to give the correct information about the FDA approved [drug](https://www.fda.gov/drugs/news-events-human-drugs/fda-approves-treatment-weight-management-patients-bardet-biedl-syndrome-aged-6-or-older).

## How to run KG-RAG

**Note: At the moment, KG-RAG is specifically designed for running prompts related to Diseases. We are actively working on improving its versatility.**

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
# GPT_API_TYPE='azure'
python -m kg_rag.rag_based_generation.GPT.text_generation -g <your favorite gpt model - "gpt-4" or "gpt-35-turbo">
# GPT_API_TYPE='openai'
python -m kg_rag.rag_based_generation.GPT.text_generation -g <your favorite gpt model - "gpt-4" or "gpt-3.5-turbo">
```

Example:

Note: The following example was run on AWS p3.8xlarge EC2 instance and using KG-RAG v0.3.0.

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/defcbff7-e777-4db6-b028-10f54c76b234" controls="controls" style="max-width: 730px;">
</video>

#### Using GPT interactive mode

This allows the user to go over each step of the process in an interactive fashion

```
# GPT_API_TYPE='azure'
python -m kg_rag.rag_based_generation.GPT.text_generation -i True -g <your favorite gpt model - "gpt-4" or "gpt-35-turbo">
# GPT_API_TYPE='openai'
python -m kg_rag.rag_based_generation.GPT.text_generation -i True -g <your favorite gpt model - "gpt-4" or "gpt-3.5-turbo">
```

#### Using Llama
Note: If you haven't downloaded Llama during [setup](https://github.com/BaranziniLab/KG_RAG#step-5-run-the-setup-script) step, then when you run the following, it may take sometime since it will download the model first.

```
python -m kg_rag.rag_based_generation.Llama.text_generation -m <method-1 or method2, if nothing is mentioned it will take 'method-1'>
```

Example:

Note: The following example was run on AWS p3.8xlarge EC2 instance and using KG-RAG v0.3.0.

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/94bda923-dafb-451a-943a-1d7c65f3ffd4" controls="controls" style="max-width: 730px;">
</video>

#### Using Llama interactive mode

This allows the user to go over each step of the process in an interactive fashion

```
python -m kg_rag.rag_based_generation.Llama.text_generation -i True -m <method-1 or method2, if nothing is mentioned it will take 'method-1'>
```

### Command line arguments for KG-RAG

| Argument | Default Value         | Definition                                               | Allowed Options                     | Notes                                                            |
|----------|-----------------|----------------------------------------------------------|------------------------------------|------------------------------------------------------------------|
| -g       | gpt-35-turbo    | GPT model selection                                      | gpt models provided by OpenAI     | Use only for GPT models                                          |
| -i       | False           | Flag for interactive mode (shows step-by-step)           | True or False                      | Can be used for both GPT and Llama models                        |
| -e       | False           | Flag for showing evidence of association from the graph | True or False                      | Can be used for both GPT and Llama models                        |
| -m       | method-1        | Which tokenizer method to use                            | method-1 or method-2. method-1 uses 'AutoTokenizer' and method-2 uses 'LlamaTokenizer' and with an additional 'legacy' flag set to False while initiating the tokenizer              | Use only for Llama models|


## BiomixQA: Benchmark dataset

BiomixQA is a curated biomedical question-answering dataset utilized to validate KG-RAG framework across different LLMs. This consists of:

- Multiple Choice Questions (MCQ)
- True/False Questions

The diverse nature of questions in this dataset, spanning multiple choice and true/false formats, along with its coverage of various biomedical concepts, makes it particularly suitable to support research and development in biomedical natural language processing, knowledge graph reasoning, and question-answering systems.

This dataset is currently hosted in Hugging Face and you can find it [here](https://huggingface.co/datasets/kg-rag/BiomixQA).

It’s easy to get started—just three lines of Python to load the dataset:

```
from datasets import load_dataset

# For MCQ data
mcq_data = load_dataset("kg-rag/BiomixQA", "mcq")

# For True/False data
tf_data = load_dataset("kg-rag/BiomixQA", "true_false")
```


## Citation

```
@article{soman2023biomedical,
  title={Biomedical knowledge graph-enhanced prompt generation for large language models},
  author={Soman, Karthik and Rose, Peter W and Morris, John H and Akbas, Rabia E and Smith, Brett and Peetoom, Braian and Villouta-Reyes, Catalina and Cerono, Gabriel and Shi, Yongmei and Rizk-Jackson, Angela and others},
  journal={arXiv preprint arXiv:2311.17330},
  year={2023}
}
```














