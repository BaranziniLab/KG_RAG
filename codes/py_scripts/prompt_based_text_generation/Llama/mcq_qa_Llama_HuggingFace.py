from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
import torch
import pandas as pd
import os
import time
import sys


MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
BRANCH_NAME="main"
QUESTION_PATH="/data1/somank/llm_data/analysis/test_questions_two_hop_mcq.csv"
SAVE_PATH="/data1/somank/llm_data/analysis"
CACHE_DIR="/data1/somank/llm_data/llm_models/huggingface"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

SYSTEM_PROMPT = """
You are an expert biomedical researcher. Please provide your answer in the following JSON format for the Question asked:
{{
  "answer": <correct answer>
}}
"""
INSTRUCTION = "Question: {question}"



def main():    
    llm = model(MODEL_NAME, BRANCH_NAME)               
    template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    start_time = time.time()
    SAVE_NAME = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_prompt_based_two_hop_mcq_response.csv"
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    for index, row in question_df.iterrows():
        question = row["text"]
        output = llm_chain.run(question)
        answer_list.append((row["text"], row["label"], output))
    answer_df = pd.DataFrame(answer_list, columns=["question", "label", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, SAVE_NAME), index=False, header=True)    
    print("Completed in {} min".format((time.time()-start_time)/60))



def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def model(MODEL_NAME, BRANCH_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                             cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,                                             
                                        device_map='auto',
                                        torch_dtype=torch.float16,
                                        revision=BRANCH_NAME,
                                        cache_dir=CACHE_DIR
                                        )
    # gptq_config = GPTQConfig(bits=4, group_size=64, desc_act=True)
    pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                max_new_tokens = 512,
                do_sample = True
                )    
    llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":0, "top_p":1})
    return llm

# llm = model(MODEL_NAME, BRANCH_NAME)               
# template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
# prompt = PromptTemplate(template=template, input_variables=["question"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# question = input("Enter your question : ")
# output = llm_chain.run(question)
# print(output)


if __name__ == "__main__":
    main()
