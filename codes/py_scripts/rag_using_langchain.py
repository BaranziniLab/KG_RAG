from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
import sys

MODEL_PATH = sys.argv[1]
VECTOR_DB_PATH = sys.argv[2]
SENTENCE_EMBEDDING_MODEL = sys.argv[3]
QUESTION_TYPE = sys.argv[4]

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




prompt = """
query : {}
Answer the above query in the following format:
{{
answer : [answer 1, answer 2, answer 3 etc]
}}
If you don't know the answer, report it as:
{{
answer : Don't know
}}
""".format(question)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True
)
result = qa_chain(prompt)
print(result["source_documents"])



# template = """Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer.   
# {context}
# Question: {question}
# Helpful Answer:"""

# QA_CHAIN_PROMPT = PromptTemplate(
#     input_variables=["context", "question"],
#     template=template,
# )

# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 40, "lambda_mult":0.5, "k":25}),
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
#     return_source_documents=True
# )

# result = qa_chain({"query": question})

