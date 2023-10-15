from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)


streamer = TextStreamer(tokenizer)

pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                max_length = 30,
                streamer=streamer
                )

question = input("Enter you statement : ")
# pipe(question, max_length=30, num_return_sequences=1, do_sample=True)

llm = HuggingFacePipeline(pipeline = pipe,
                          model_kwargs = {"temperature":0, "top_p":1})

llm(question)

