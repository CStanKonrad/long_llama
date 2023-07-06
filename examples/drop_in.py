from transformers import LlamaTokenizer, LlamaForCausalLM
from .checkpoints import *
import torch

MODEL_PATH = LONGLLAMA3B_PATH

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
