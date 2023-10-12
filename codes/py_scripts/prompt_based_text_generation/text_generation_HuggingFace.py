from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
import torch


MODEL_NAME = "TheBloke/Llama-2-13B-chat-GPTQ"
BRANCH_NAME = "gptq-4bit-64g-actorder_True"
CACHE_DIR = "/data1/somank/llm_data/llm_models/huggingface"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
SYSTEM_PROMPT = """
You are a biomedical researcher. Answer the Question asked. If you don't know the answer, report as "I don't know", don't try to make up an answer.
"""
INSTRUCTION = "Question: {question}"


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template




tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                         use_auth_token=True,
                                         cache_dir=CACHE_DIR
                                         )
# gptq_config = GPTQConfig(bits=4, group_size=64, desc_act=True)
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
                          model_kwargs = {'temperature':0})


template = get_prompt(INSTRUCTION, SYSTEM_PROMPT)
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = input("Enter your question : ")
output = llm_chain.run(question)
print(output)
