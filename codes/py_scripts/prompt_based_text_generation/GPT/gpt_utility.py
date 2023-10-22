import openai
import os
from dotenv import load_dotenv, find_dotenv


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
