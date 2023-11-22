import os
from kg_rag.utility import config_data

def download_llama():
    from kg_rag.utility import llama_model
    try:
        llama_model(config_data["LLAMA_MODEL_NAME"], config_data["LLAMA_MODEL_BRANCH"], config_data["LLM_CACHE_DIR"])
        print("Model is successfully downloaded to the provided cache directory!")
    except:
        print("Model is not downloaded! Make sure the above mentioned conditions are satisfied")
        

print("")
print("Starting to set up KG-RAG ...")
print("")

user_input = input("Did you update the config.yaml file with all necessary configurations (such as GPT .env path, vectorDB file paths, other file paths)? Enter Y or N: ")
print("")
if user_input == "Y":
    print("Checking disease vectorDB ...")
    try:
        if os.path.exists(config_data["VECTOR_DB_PATH"]):
            print("vectorDB already exists!")
        else:
            print("Creating vectorDB ...")
            from kg_rag.vectorDB.create_vectordb import create_vectordb
            create_vectordb()
    except:
        print("Double check the path that was given in VECTOR_DB_PATH of config.yaml file.")

    print("")
    user_input_1 = input("Do you want to install Llama model? Enter Y or N: ")
    if user_input_1 == "Y":
        user_input_2 = input("Did you update the config.yaml file with proper configuration for downloading Llama model? Enter Y or N: ")
        if user_input_2 == "Y":
            user_input_3 = input("Are you using official Llama model from Meta? Enter Y or N: ")
            if user_input_3 == "Y":
                user_input_4 = input("Did you get access to use the model? Enter Y or N: ")
                if user_input_4 == "Y":
                    download_llama()    
                else:
                    print("Aborting!")
            else:
                download_llama()
        else:
            print("Aborting!")
    else:
        print("No problem. Llama will get installed on-the-fly when you run the model for the first time.")
else:
    print("As the first step, update config.yaml file and then run this python script again.")
        
            
    