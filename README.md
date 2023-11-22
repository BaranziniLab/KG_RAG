# Knowledge Graph-based Retrieval Augmented Generation (KG-RAG)
KG-RAG is a task agnostic framework that combines the explicit knwoledge of a Knowledge Graph (KG) with the implicit knwoledge of a Large Language Model (LLM). 

The main feature of KG-RAG is that it extracts "prompt-aware context" from the KG, which is defined as: 

**the minimal context sufficient enough to respond to the user prompt.** 

Hence, the framework provides context by optimizing the input token space of the LLM.

## Example use case of KG-RAG
Following snippet shows the news from FDA website about the drug **"setmelanotide"** approved by FDA for weight management in patients with *Bardet-Biedl Syndrome*

<img src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/fc4d0b8d-0edb-461d-86c5-9d0d191bd97d" width="600" height="350">

Inputting the 


