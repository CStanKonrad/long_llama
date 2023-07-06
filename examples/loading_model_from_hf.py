import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM
from .checkpoints import *

MODEL_PATH = LONGLLAMA3B_PATH

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, trust_remote_code=True)
