from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import pipeline
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import sys

VECTOR_DB_PATH = sys.argv[1]
SENTENCE_EMBEDDING_MODEL = sys.argv[2]
MODEL_NAME = sys.argv[3]
BRANCH_NAME = sys.argv[4]
CACHE_DIR = sys.argv[5]

# "meta-llama/Llama-2-13b-chat-hf"

RETRIEVAL_SCORE_THRESH = 0.72


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


SYSTEM_PROMPT = """
You are a biomedical researcher. For answering the question at the end, you need to first read the Context provided and then answer the Question. If you don't know the answer, report as "I don't know", don't try to make up an answer.
"""
INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)

vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, 
                     embedding_function=embedding_function)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                         use_auth_token=True,
                                         cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,                                             
                                    device_map='auto',
                                    torch_dtype=torch.float16,
                                    use_auth_token=True,
                                    revision=BRANCH_NAME,
                                    cache_dir=CACHE_DIR
                                    )
streamer = TextStreamer(tokenizer)
pipe = pipeline("text-generation",
            model = model,
            tokenizer = tokenizer,
            torch_dtype = torch.bfloat16,
            device_map = "auto",
            max_new_tokens = 512,
            do_sample = True,
            top_k = 30,
            num_return_sequences = 1,
            streamer=streamer
            )

llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":0, "top_p":1})



template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)

print("")
question = input("Enter your question : ")
print("")

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

context = retrieved_context
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
output = llm_chain.run(context=context, question=question)





