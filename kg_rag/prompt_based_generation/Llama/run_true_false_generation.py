from langchain import PromptTemplate, LLMChain
from kg_rag.utility import *

QUESTION_PATH = config_data["TRUE_FALSE_PATH"]
SYSTEM_PROMPT = system_prompts["TRUE_FALSE_QUESTION_PROMPT_BASED"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]
MODEL_NAME = config_data["LLAMA_MODEL_NAME"]
BRANCH_NAME = config_data["LLAMA_MODEL_BRANCH"]
CACHE_DIR = config_data["LLM_CACHE_DIR"]


INSTRUCTION = "Question: {question}"

save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_prompt_based_one_hop_true_false_binary_response.csv"

def main():
    start_time = time.time()
    llm = llama_model(MODEL_NAME, BRANCH_NAME, CACHE_DIR)
    template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    for index, row in question_df.iterrows():
        question = row["text"]
        output = llm_chain.run(question)
        answer_list.append((row["text"], row["label"], output))
    answer_df = pd.DataFrame(answer_list, columns=["question", "label", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)  
    print("Completed in {} min".format((time.time()-start_time)/60))
    
    

if __name__ == "__main__":
    main()
