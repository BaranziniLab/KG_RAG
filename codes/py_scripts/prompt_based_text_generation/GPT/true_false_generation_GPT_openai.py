import pandas as pd
import time
from gpt_utility import *


CHAT_MODEL_ID = "gpt-35-turbo"
CHAT_DEPLOYMENT_ID = None
QUESTION_PATH = "../../../../data/benchmark_datasets/test_questions.csv"
SAVE_PATH = "../../../../data/analysis_results"

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_prompt_based_binary_response.csv"

# GPT config params
temperature = 0

if not CHAT_DEPLOYMENT_ID:
    CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID
    

system_prompt = """
You are an expert biomedical researcher. Please provide your answer in the following JSON format for the Question asked:
{{
  "answer": "True"
}}
OR
{{
  "answer": "False"
}}
"""

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    for index, row in question_df.iterrows():
        question = "Question: "+ row["text"]
        output = get_GPT_response(question, system_prompt, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=temperature)
        answer_list.append((row["text"], row["label"], output))
    answer_df = pd.DataFrame(answer_list, columns=["question", "label", "llm_answer"])
    answer_df.to_csv(os.path.join(SAVE_PATH, save_name), index=False, header=True)
    print("Completed in {} min".format((time.time()-start_time)/60))
    


if __name__ == "__main__":
    main()
    
