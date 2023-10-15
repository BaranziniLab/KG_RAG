# pip install accelerate
from transformers import AutoTokenizer, OPTForCausalLM

MODEL_NAME = "facebook/galactica-1.3b"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = OPTForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", cache_dir=CACHE_DIR)

input_text = "The Transformer architecture [START_REF]"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
