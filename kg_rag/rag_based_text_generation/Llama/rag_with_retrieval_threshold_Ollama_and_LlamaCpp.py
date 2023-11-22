from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import sys

MODEL_PATH = sys.argv[1]
VECTOR_DB_PATH = sys.argv[2]
SENTENCE_EMBEDDING_MODEL = sys.argv[3]
QUESTION_TYPE = sys.argv[4]

RETRIEVAL_SCORE_THRESH = 0.72

if QUESTION_TYPE == "cmd_prompt":
    question = input("Enter your question: ")
    print("")
else:
    question = "What compounds treat multiple sclerosis?​ State the provenance and phase of the treatment"

embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)

vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, 
                     embedding_function=embedding_function)

if MODEL_PATH == "ollama":
    from langchain.llms import Ollama
    llm = Ollama(base_url="http://localhost:11434",
                 model="llama2:13b",
                 temperature=0,
                 verbose=True,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
else:
    from langchain.llms import LlamaCpp
    llm = LlamaCpp(
                model_path=MODEL_PATH,
                temperature=0,
                top_p=1,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), 
                verbose=True)

search_result = vectorstore.similarity_search_with_score(question, k=10000)
score_range = (search_result[-1][-1] - search_result[0][-1]) / (search_result[-1][-1] + search_result[0][-1])
thresh = RETRIEVAL_SCORE_THRESH*score_range
retrieved_context = ""
for item in search_result:
    item_score = (search_result[-1][-1] - item[-1]) / (search_result[-1][-1] + item[-1])
    if item_score < thresh:
        break
    retrieved_context += item[0].page_content
    retrieved_context += "\n"

# prompt = """
# Use the following pieces of context to answer the question at the end. 
# Context: {}
# Question : {}
# Based on the context provided, answer the above Question in the following format:
# {{
# answer : [answer 1, answer 2, answer 3 etc]
# }}
# If you don't know the answer, report it as:
# {{
# answer : Don't know
# }}
# """.format(retrieved_context, question)

prompt = """
Use the following pieces of context to answer the question at the end as True or False. 
Context: {}
Question : {}
Based on the context provided, answer the above Question in the following format:
{{
answer : <True> or <False>
}}
If you don't know the answer, report it as:
{{
answer : Don't know
}}
""".format(retrieved_context, question)

llm_response = llm(prompt)

