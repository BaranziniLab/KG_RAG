# Knowledge Graph-based Retrieval Augmented Generation (KG-RAG)

## Table of Contents
[What is KG-RAG](https://github.com/BaranziniLab/KG_RAG#what-is-kg-rag)

[Example use case of KG-RAG](https://github.com/BaranziniLab/KG_RAG#example-use-case-of-kg-rag)

 - [Prompting GPT without KG-RAG](https://github.com/BaranziniLab/KG_RAG#prompting-gpt-35-turbo-model-without-kg-rag-about-the-above-mentioned-fda-approved-drug)
  
 - [Prompting GPT with KG-RAG](https://github.com/BaranziniLab/KG_RAG#prompting-gpt-35-turbo-model-with-kg-rag-about-the-above-mentioned-fda-approved-drug)

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








