from kg_rag.utility import *
import sys


CHAT_MODEL_ID = sys.argv[1]

SYSTEM_PROMPT = system_prompts["PROMPT_BASED_TEXT_GENERATION"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]

CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID


def main():
    question = input("Enter your question : ")    
    print("Here is the prompt-based answer:")
    print("")
    output = get_GPT_response(question, SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=TEMPERATURE)
    stream_out(output)
    
def stream_out(output):
    CHUNK_SIZE = 10
    SLEEP_TIME = 0.1
    for i in range(0, len(output), CHUNK_SIZE):
        print(output[i:i+CHUNK_SIZE], end='')
        sys.stdout.flush()
        time.sleep(SLEEP_TIME)
    
    
    
if __name__ == "__main__":
    main()


