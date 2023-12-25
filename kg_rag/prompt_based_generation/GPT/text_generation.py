from kg_rag.utility import *
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-g', type=str, default='gpt-35-turbo', help='GPT model selection')
args = parser.parse_args()

CHAT_MODEL_ID = args.g

SYSTEM_PROMPT = system_prompts["PROMPT_BASED_TEXT_GENERATION"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]

CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID


def main():
    print(" ")
    question = input("Enter your question : ")    
    print("Here is the prompt-based answer:")
    print("")
    output = get_GPT_response(question, SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=TEMPERATURE)
    stream_out(output)
    

    
    
    
if __name__ == "__main__":
    main()


