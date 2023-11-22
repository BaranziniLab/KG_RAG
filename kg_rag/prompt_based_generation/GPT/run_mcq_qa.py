from kg_rag.utility import *
import sys


CHAT_MODEL_ID = sys.argv[1]

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION_PROMPT_BASED"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]

CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_prompt_based_response_for_two_hop_mcq_from_monarch_and_robokop.csv"


def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    for index, row in question_df.iterrows():
        question = "Question: "+ row["text"]
        output = get_GPT_response(question, SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=TEMPERATURE)
        answer_list.append((row["text"], row["correct_node"], output))                  
    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)
    print("Completed in {} min".format((time.time()-start_time)/60))
    
    
if __name__ == "__main__":
    main()