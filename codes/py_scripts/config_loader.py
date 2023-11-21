import yaml

with open('../config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
    
with open('../file_paths.yaml', 'r') as f:
    file_path_data = yaml.safe_load(f)
    
with open('../system_prompts.yaml', 'r') as f:
    system_prompts = yaml.safe_load(f)
    
__all__ = [
    'config_data',
    'file_path_data',
    'system_prompts'
]