import yaml
import os

with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
        
with open('system_prompts.yaml', 'r') as f:
    system_prompts = yaml.safe_load(f)
    
if 'GPT_CONFIG_FILE' in config_data:
    config_data['GPT_CONFIG_FILE'] = config_data['GPT_CONFIG_FILE'].replace('$HOME', os.environ['HOME'])

    
__all__ = [
    'config_data',
    'system_prompts'
]