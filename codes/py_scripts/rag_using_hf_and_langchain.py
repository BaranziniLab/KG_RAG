from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import pipeline
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GGML"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          use_auth_token=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True
                                             )


pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                max_new_tokens = 512,
                do_sample = True,
                top_k = 30,
                num_return_sequences = 1,
                eos_token_id = tokenizer.eos_token_id
                )

llm = HuggingFacePipeline(pipeline = pipe, 
                          callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                          verbose=True,
                          model_kwargs = {'temperature':0})



system_prompt = "You are an advanced assistant that excels at translation. "
instruction = "Answer the question asked:\n\n {question}"
template = get_prompt(instruction, system_prompt)

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = input("Enter your question : ")
# output = llm_chain.run(question)
llm(question)
# output = llm(question)
# print(output)





