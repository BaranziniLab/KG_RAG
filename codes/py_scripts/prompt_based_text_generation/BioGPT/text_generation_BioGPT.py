from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import pipeline, BioGptTokenizer, BioGptForCausalLM, TextStreamer
import torch

MODEL_NAME = "microsoft/BioGPT-Large"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"

model = BioGptForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer = BioGptTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)


streamer = TextStreamer(tokenizer)

pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                streamer=streamer,
                temperature=0
                )

question = input("Enter you statement : ")
pipe(question, max_length=20, num_return_sequences=1, do_sample=True)