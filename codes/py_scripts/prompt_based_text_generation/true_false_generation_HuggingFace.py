from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
import torch
import pandas as pd
import os
import time
import sys


MODEL_NAME = sys.argv[1]
BRANCH_NAME = sys.argv[2]
QUESTION_PATH = sys.argv[3]
SAVE_PATH = sys.argv[4]
STREAM = sys.argv[5]



# MODEL_NAME = "TheBloke/Llama-2-13B-chat-GPTQ"
# BRANCH_NAME = "gptq-4bit-64g-actorder_True"

stream_dict = {
    "True" : True,
    "False" : False
}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
SYSTEM_PROMPT = """
You are a biomedical researcher. Answer the given Question as either True or False. Don't give any other explanations. If you don't know the answer, report as "Don't know", don't try to make up an answer. Provide the answer in the following format:
{{answer : <True> or <False> or <Don't know>}}
"""
INSTRUCTION = "Question: {question}"


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def parse_response(response):
    return response.split("{answer: ")[-1].split("}")[0]

def model(MODEL_NAME, BRANCH_NAME, stream=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                             use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,                                             
                                        device_map='auto',
                                        torch_dtype=torch.float16,
                                        use_auth_token=True,
                                        revision=BRANCH_NAME                                                
                                        )
    # gptq_config = GPTQConfig(bits=4, group_size=64, desc_act=True)
    if stream:
        streamer = TextStreamer(tokenizer)
        pipe = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    max_new_tokens = 512,
                    do_sample = True,
                    top_k = 30,
                    num_return_sequences = 1,
                    streamer=streamer
                    )

    else:
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
    llm = model(MODEL_NAME, BRANCH_NAME, stream=stream_dict[STREAM])               
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
            answer_list.append((row["text"], row["label"], parse_response(output)))
        answer_df = pd.DataFrame(answer_list, columns=["question", "label", "llm_answer"])
        answer_df.to_csv(os.path.join(SAVE_PATH, SAVE_NAME), index=False, header=True)    
        print("Completed in {} min".format((time.time()-start_time)/60))
    else:
        question = input("Enter your question : ")
        output = llm_chain.run(question)
        print(parse_response(output))

if __name__ == "__main__":
    main()
