from langchain.llms import HuggingFacePipeline
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
import sys

MODEL_PATH = sys.argv[1]
VECTOR_DB_PATH = sys.argv[2]
SENTENCE_EMBEDDING_MODEL = sys.argv[3]
QUESTION = sys.argv[4]
  
MODEL_NAME = "Yukang/Llama-2-13b-chat-longlora-32k-sft"

llm = HuggingFacePipeline.from_model_id(
    model_id=MODEL_NAME,
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 4096},
)

embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)

vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, 
                     embedding_function=embedding_function)

template = """Use the following pieces of context to answer the question at the end and also to return the provenance. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.   
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 40, "lambda_mult":0.5, "k":2}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

result = qa_chain({"query": QUESTION})
