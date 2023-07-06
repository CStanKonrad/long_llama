import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM
from .checkpoints import *

MODEL_PATH = LONGLLAMA3B_PATH


tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, trust_remote_code=True)

prompt = "My name is Julien and I like to"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model(input_ids=input_ids)

torch.manual_seed(60)
generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    num_beams=1,
    last_context_length=1792,
    do_sample=True,
    temperature=1.0,
)
print(tokenizer.decode(generation_output[0]))
