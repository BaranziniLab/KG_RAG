from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoTokenizer, OPTForCausalLM, pipeline, TextStreamer
import torch
import pandas as pd
import os
import time
import sys


MODEL_NAME = "facebook/galactica-1.3b"
BRANCH_NAME = "main"
CACHE_DIR = "/data/somank/llm_data/llm_models/huggingface"
QUESTION_PATH = None

template = """
Question: {question} If you don't know the answer, report as "Don't know", don't try to make up an answer\n\nAnswer:
"""


def model(MODEL_NAME, BRANCH_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                             cache_dir=CACHE_DIR)
    model = OPTForCausalLM.from_pretrained(MODEL_NAME,                                             
                                        device_map='auto',
                                        torch_dtype=torch.float16,
                                        revision=BRANCH_NAME,
                                        cache_dir=CACHE_DIR
                                        )
    pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                max_new_tokens = 512,
                do_sample = True,
                top_k = 30,
                num_return_sequences = 1
                )

    
    llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":0, "top_p":1})
    return llm


def main():    
    llm = model(MODEL_NAME, BRANCH_NAME)               
    template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    if QUESTION_PATH:
        start_time = time.time()
        SAVE_NAME = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_prompt_based_response.csv"
        question_df = pd.read_csv(QUESTION_PATH)
        answer_list = []
        for index, row in question_df.iterrows():
            question = row["text"]
            output = llm_chain.run(question)
            answer_list.append((row["text"], row["label"], output))
        answer_df = pd.DataFrame(answer_list, columns=["question", "label", "llm_answer"])
        answer_df.to_csv(os.path.join(SAVE_PATH, SAVE_NAME), index=False, header=True)    
        print("Completed in {} min".format((time.time()-start_time)/60))
    else:
        question = input("Enter your question : ")
        output = llm_chain.run(question)
        print(output)

if __name__ == "__main__":
    main()