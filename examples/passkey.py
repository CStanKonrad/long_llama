from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch
from .utils.landmark_prompt import generate_prompt_landmark
from .checkpoints import *



MODEL_PATH = LONGLLAMA3B_PATH
print(f"Loading tokenizer {MODEL_PATH}")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

print(f"Loading model {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32, trust_remote_code=True, mem_attention_grouping=(1, 2048)
)

model.eval()


def passkey_retrieval_test(n_garbage=60000, seed=555):

  prompt, answer = generate_prompt_landmark(n_garbage, seed)
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  input_ids = input_ids
  print(f"Prompt has {input_ids.shape[-1]} tokens")

  answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS
  generation_output = model.generate(
      input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, last_context_length=1024
  )

  model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()

  is_correct = (model_answer == answer_ids[0]).all().item()
  print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
  print(f"The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
  return is_correct

passkey_retrieval_test(30000, 555)