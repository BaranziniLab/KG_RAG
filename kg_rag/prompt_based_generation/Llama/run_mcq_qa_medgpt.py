from langchain import PromptTemplate, LLMChain
from kg_rag.utility import *


QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION_PROMPT_BASED"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]
MODEL_NAME = 'PharMolix/BioMedGPT-LM-7B'
BRANCH_NAME = 'main'
CACHE_DIR = config_data["LLM_CACHE_DIR"]



save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_prompt_based_mcq_from_monarch_and_robokop_response.csv"

INSTRUCTION = "Question: {question}"


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
        answer_list.append((row["text"], row["correct_node"], output))
    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)
    print("Completed in {} min".format((time.time()-start_time)/60))
    
    

if __name__ == "__main__":
    main()
