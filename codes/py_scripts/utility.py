import pandas as pd
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
