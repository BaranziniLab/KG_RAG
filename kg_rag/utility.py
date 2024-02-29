import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory
import json
import openai
import os
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
from dotenv import load_dotenv, find_dotenv
import torch
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
from kg_rag.config_loader import *
import ast
import requests

memory = Memory("cachegpt", verbose=0)

# Config openai library
config_file = config_data['GPT_CONFIG_FILE']
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = config_data['GPT_API_TYPE']
openai.api_key = api_key
if resource_endpoint:
    openai.api_base = resource_endpoint
if api_version:
    openai.api_version = api_version

torch.cuda.empty_cache()
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    if params:
        return requests.get(uri, params=params)
    else:
        return requests.get(uri)

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def get_context_using_spoke_api(node_value):
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]
    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level']
    }
    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()
    nbr_nodes = []
    nbr_edges = []
    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)                    
                except:
                    try:                    
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:                                                    
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = map(lambda x:"pubmedId:"+x, pmid_list)
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:                                
                        provenance = "SPOKE-KG"     
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append((item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:,"node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name":"source"})
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:,"node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name":"target"})
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x:x.split("_")[0])
    merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + ". "
    context = merge_2.context.str.cat(sep=' ')
    context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + node_context[0]["data"]["properties"]["source"] + "."
    return context, merge_2
        
#         if edge_evidence:
#             merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + " and attributes associated with this association is in the following JSON format:\n " + merge_2.evidence.astype('str') + "\n\n"
#         else:
#             merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + ". "
#         context = merge_2.context.str.cat(sep=' ')
#         context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + node_context[0]["data"]["properties"]["source"] + "."
#     return context



def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + system_prompt + instruction + E_INST
    return prompt_template

def llama_model(model_name, branch_name, cache_dir, temperature=0, top_p=1, max_new_tokens=512, stream=False, method='method-1'):
    if method == 'method-1':
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                 revision=branch_name,
                                                 cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name,                                             
                                            device_map='auto',
                                            torch_dtype=torch.float16,
                                            revision=branch_name,
                                            cache_dir=cache_dir
                                            )
    elif method == 'method-2':
        import transformers
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, 
                                                                revision=branch_name, 
                                                                cache_dir=cache_dir, 
                                                                legacy=False)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, 
                                                              device_map='auto', 
                                                              torch_dtype=torch.float16, 
                                                              revision=branch_name, 
                                                              cache_dir=cache_dir)        
    if not stream:
        pipe = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    max_new_tokens = max_new_tokens,
                    do_sample = True
                    )
    else:
        streamer = TextStreamer(tokenizer)
        pipe = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    max_new_tokens = max_new_tokens,
                    do_sample = True,
                    streamer=streamer
                    )        
    llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":temperature, "top_p":top_p})
    return llm



@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    # print('Calling OpenAI...')
    response = openai.ChatCompletion.create(
        temperature=temperature,
        deployment_id=chat_deployment_id,
        model=chat_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
    )
    if 'choices' in response \
       and isinstance(response['choices'], list) \
       and len(response) >= 0 \
       and 'message' in response['choices'][0] \
       and 'content' in response['choices'][0]['message']:
        return response['choices'][0]['message']['content']
    else:
        return 'Unexpected response'

@memory.cache
def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    return fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature)


def stream_out(output):
    CHUNK_SIZE = int(round(len(output)/50))
    SLEEP_TIME = 0.1
    for i in range(0, len(output), CHUNK_SIZE):
        print(output[i:i+CHUNK_SIZE], end='')
        sys.stdout.flush()
        time.sleep(SLEEP_TIME)
    print("\n")

def get_gpt35():
    chat_model_id = 'gpt-35-turbo' if openai.api_type == 'azure' else 'gpt-3.5-turbo'
    chat_deployment_id = chat_model_id if openai.api_type == 'azure' else None
    return chat_model_id, chat_deployment_id

def disease_entity_extractor(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    resp = get_GPT_response(text, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    
def disease_entity_extractor_v2(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    prompt_updated = system_prompts["DISEASE_ENTITY_EXTRACTION"] + "\n" + "Sentence : " + text
    resp = get_GPT_response(prompt_updated, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)

def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, api=True):
    entities = disease_entity_extractor_v2(question)
    node_hits = []
    if entities:
        max_number_of_high_similarity_context_per_node = int(context_volume/len(entities))
        for entity in entities:
            node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
            node_hits.append(node_search_result[0][0].page_content)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node_name in node_hits:
            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            else:
                node_context,context_table = get_context_using_spoke_api(node_name)
            node_context_list = node_context.split(". ")        
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]            
            if edge_evidence:
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
                node_context_extracted = context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context)
                node_context_extracted += ". "
        return node_context_extracted
    else:
        node_hits = vectorstore.similarity_search_with_score(question, k=5)
        max_number_of_high_similarity_context_per_node = int(context_volume/5)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node in node_hits:
            node_name = node[0].page_content
            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            else:
                node_context, context_table = get_context_using_spoke_api(node_name)
            node_context_list = node_context.split(". ")        
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
            if edge_evidence:
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
                node_context_extracted = context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context)
                node_context_extracted += ". "
        return node_context_extracted
    
    
def interactive(question, vectorstore, node_context_df, embedding_function_for_context_retrieval, llm_type, edge_evidence, api=True, llama_method="method-1"):
    print(" ")
    input("Press enter for Step 1 - Disease entity extraction using GPT-3.5-Turbo")
    print("Processing ...")
    entities = disease_entity_extractor_v2(question)
    max_number_of_high_similarity_context_per_node = int(config_data["CONTEXT_VOLUME"]/len(entities))
    print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
    print(" ")
    
    input("Press enter for Step 2 - Match extracted Disease entity to SPOKE nodes")
    print("Finding vector similarity ...")
    node_hits = []
    for entity in entities:
        node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
        node_hits.append(node_search_result[0][0].page_content)
    print("Matched entities from SPOKE = '{}'".format(", ".join(node_hits)))
    print(" ")
    
    input("Press enter for Step 3 - Context extraction from SPOKE")
    node_context = []
    for node_name in node_hits:
        if not api:
            node_context.append(node_context_df[node_context_df.node_name == node_name].node_context.values[0])
        else:
            context, context_table = get_context_using_spoke_api(node_name)
            node_context.append(context)
    print("Extracted Context is : ")
    print(". ".join(node_context))
    print(" ")

    input("Press enter for Step 4 - Context pruning")
    question_embedding = embedding_function_for_context_retrieval.embed_query(question)
    node_context_extracted = ""
    for node_name in node_hits:
        if not api:
            node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        else:
            node_context, context_table = get_context_using_spoke_api(node_name)
        node_context_list = node_context.split(". ")        
        node_context_embeddings = embedding_function_for_context_retrieval.embed_documents(node_context_list)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities], config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
        high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"]]
        if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
            high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
        if edge_evidence:
            context_table = context_table[context_table.context.isin(high_similarity_context)]
            print(high_similarity_context[0])
            print(context_table)
            context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
            node_context_extracted = context_table.context.str.cat(sep=' ')
        else:
            node_context_extracted += ". ".join(high_similarity_context)
            node_context_extracted += ". "
    print("Pruned Context is : ")
    print(node_context_extracted)
    print(" ")
    
    input("Press enter for Step 5 - LLM prompting")
    print("Prompting ", llm_type)
    if llm_type == "llama":
        from langchain import PromptTemplate, LLMChain
        template = get_prompt("Context:\n\n{context} \n\nQuestion: {question}", system_prompts["KG_RAG_BASED_TEXT_GENERATION"])
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = llama_model(config_data["LLAMA_MODEL_NAME"], config_data["LLAMA_MODEL_BRANCH"], config_data["LLM_CACHE_DIR"], stream=True, method=llama_method) 
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        output = llm_chain.run(context=node_context_extracted, question=question)
    elif "gpt" in llm_type:
        enriched_prompt = "Context: "+ node_context_extracted + "\n" + "Question: " + question
        output = get_GPT_response(enriched_prompt, system_prompts["KG_RAG_BASED_TEXT_GENERATION"], llm_type, llm_type, temperature=config_data["LLM_TEMPERATURE"])
        stream_out(output)