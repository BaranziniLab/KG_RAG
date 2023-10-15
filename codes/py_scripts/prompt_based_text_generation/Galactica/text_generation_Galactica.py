
from transformers import AutoTokenizer, OPTForCausalLM, pipeline, TextStreamer
import torch

MODEL_NAME = "facebook/galactica-1.3b"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = OPTForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", cache_dir=CACHE_DIR)

streamer = TextStreamer(tokenizer)

pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                max_length = 30,
                streamer=streamer
                )

input_text = input("Enter your text : ")
pipe(input_text, max_length=30, num_return_sequences=1, do_sample=True)

# input_text = "The Transformer architecture [START_REF]"

# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))
