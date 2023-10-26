import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import openai
import os
from dotenv import load_dotenv, find_dotenv

def create_mcq(df, source_column, target_column, node_type, predicate):
    disease_pairs = df[source_column].unique()
    disease_pairs = [(disease1, disease2) for disease1 in disease_pairs for disease2 in disease_pairs if disease1 != disease2]

    new_data = []

    #For each source pair, find a common target and 4 negative samples
    for disease1, disease2 in disease_pairs:
        common_gene = set(df[df[source_column] == disease1][target_column]).intersection(set(df[df[source_column] == disease2][target_column]))
        common_gene = list(common_gene)[0] if common_gene else None
        # Get 4 random negative samples
        negative_samples = df[(df[source_column] != disease1) & (df[source_column] != disease2)][target_column].sample(4).tolist()
        new_data.append(((disease1, disease2), common_gene, negative_samples))

    new_df = pd.DataFrame(new_data, columns=["disease_pair", "correct_node", "negative_samples"])
    new_df.dropna(subset = ["correct_node"], inplace=True)
    new_df.loc[:, "disease_1"] = new_df["disease_pair"].apply(lambda x: x[0])
    new_df.loc[:, "disease_2"] = new_df["disease_pair"].apply(lambda x: x[1])
    new_df.negative_samples = new_df.negative_samples.apply(lambda x:", ".join(x[0:4]))
    new_df.loc[:, "text"] = "Out of the given list, which " + node_type + " " + predicate + " " + new_df.disease_1 + " and " + new_df.disease_2 + ". Given list is: " + new_df.correct_node + ", " + new_df.negative_samples
    return new_df

# Config openai library
config_file = os.path.join(os.path.expanduser('~'), '.gpt_config.env')
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = "azure"
openai.api_key = api_key
openai.api_base = resource_endpoint
openai.api_version = api_version


def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    try:
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
    except:
        return None


def disease_entity_extractor(text):
    chat_deployment_id = 'gpt-35-turbo'
    chat_model_id = 'gpt-35-turbo'
    temperature = 0
    system_prompt = """
    You are an expert disease entity extractor from a sentence and report it as JSON in the following format:
    Diseases : <List of extracted entities>
    """
    resp = get_GPT_response(text, system_prompt, chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    
def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold):
    entities = disease_entity_extractor(question)
    node_hits = []
    if entities:
        max_number_of_high_similarity_context_per_node = int(context_volume/len(entities))
        for entity in entities:
            node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
            node_hits.append(node_search_result[0][0].page_content)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node_name in node_hits:
            node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            node_context_list = node_context.split(". ")        
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
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
            node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            node_context_list = node_context.split(". ")        
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
            node_context_extracted += ". ".join(high_similarity_context)
            node_context_extracted += ". "
        return node_context_extracted
    