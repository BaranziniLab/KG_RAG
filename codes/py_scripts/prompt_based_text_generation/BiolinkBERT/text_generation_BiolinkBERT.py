from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import pipeline, AutoTokenizer, AutoModel, TextStreamer
import torch

MODEL_NAME = "michiyasunaga/BioLinkBERT-base"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"

model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)


streamer = TextStreamer(tokenizer)

pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                streamer=streamer,
                temperature=0.1
                )

question = input("Enter you statement : ")
pipe(question, max_length=100, num_return_sequences=1, do_sample=True)