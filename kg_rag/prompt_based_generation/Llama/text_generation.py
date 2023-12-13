from langchain import PromptTemplate, LLMChain
from kg_rag.utility import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, default='method-1', help='Method to choose for Llama model')
args = parser.parse_args()

METHOD = args.m


SYSTEM_PROMPT = system_prompts["PROMPT_BASED_TEXT_GENERATION"]
MODEL_NAME = config_data["LLAMA_MODEL_NAME"]
BRANCH_NAME = config_data["LLAMA_MODEL_BRANCH"]
CACHE_DIR = config_data["LLM_CACHE_DIR"]



INSTRUCTION = "Question: {question}"


def main():
    llm = llama_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR, stream=True, method=METHOD)
    template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["question"])    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(" ")
    question = input("Enter your question : ")
    print("Here is the prompt-based answer:")
    print("")
    output = llm_chain.run(question)



if __name__ == "__main__":
    main()
