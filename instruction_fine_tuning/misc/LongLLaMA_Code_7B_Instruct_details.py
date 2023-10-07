datasets = (
    "TIGER-Lab/MathInstruct,Open-Orca/OpenOrca,Open-Orca/OpenOrca,zetavg/ShareGPT-Processed,zetavg/ShareGPT-Processed",
)
revisions = (
    "95969ba885cf13d89c1fc0990c1a246a46e388d6,f0823c7ffc48c9d33a42f16cf0b885fed4a7d0a1,f0823c7ffc48c9d33a42f16cf0b885fed4a7d0a1,15968d6dfa02529988da12e382af3ab7c857ebcd,15968d6dfa02529988da12e382af3ab7c857ebcd",
)
probs = [0.71, 0.16, 0.1, 0.02, 0.01]

BASE_MATHINSTRUCT_FILTER = ""
BASE_ORCA_FILTER = "question<LENLT>5000"
LONG_ORCA_FILTER = "question<LENGT>5000<;>question<LENLT>12000<;>response<M>(?i)^((?!(does not mention)|(it is not possible to)|(no answer\.$)|(Insufficient information to provide an answer\.$)).)*$"
BASE_CHAT_FILTER = "lang<M>^en$<;>conversations.<int>*.value<LENLT>6000<;>conversations.<int>*^.value<M>(?i)^((?!((as a|I am).{0,32}language model)|(openai)|(against my guidelines)|(I apologize)).)*$<;>conversations.<int>*^.value<M>^((?!DAN).)*$"
LONG_CHAT_FILTER = "lang<M>^en$<;>conversations.<int>*.value<LENGT>6000<;>conversations.<int>*.value<LENLT>12000<;>conversations.<int>*^.value<M>(?i)^((?!((as a|I am).{0,32}language model)|(openai)|(against my guidelines)|(I apologize)).)*$<;>conversations.<int>*^.value<M>^((?!DAN).)*$"

filtering = "<,>".join(
    [BASE_MATHINSTRUCT_FILTER, BASE_ORCA_FILTER, LONG_ORCA_FILTER, BASE_CHAT_FILTER, LONG_CHAT_FILTER]
)

{
    "data_type": "instructions,instructions,instructions,chat,chat",
    "data_path": datasets,
    "data_revision": revisions,
    "dataset_split": "train,train,train,train,train",
    "data_proportions": probs,
    "prompt_field": None,
    "pre_question_text": "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can.\n\n",
    "question_field": "instruction,question,question",
    "post_question_text": "\nAnswer: <,> <,> ",
    "response_field": "output,response,response",
    "chat_conversations_field": "conversations",
    "chat_data_field": "markdown",
    "chat_source_name_field": "from",
    "chat_model_source_name": "gpt",
    "chat_initial_prompt": "A chat between a user (denoted as USER:) and an artificial intelligence assistant (denoted as ASSISTANT:). The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n",
    "chat_model_response_prefix": "\nASSISTANT: ",
    "chat_human_response_prefix": "\nUSER: ",
    "max_input_length": 3072,
    "max_output_length": 3072,
    "max_total_length": 3072,
    "always_pad": True,
    "random_pad": True,
    "batch_size": 128,
    "last_context_length": 1536,
    "optimizer.accumulate_gradient_steps": 8,
    "total_steps": 9000,
    "dtype": "bf16",
    "mesh_dim": "1,16,8",
    "load_llama_config": "7b_FoT",
    "optimizer.type": "adamw",
    "optimizer.adamw_optimizer.lr": 1.2e-5,
    "optimizer.adamw_optimizer.end_lr": 0,
    "optimizer.adamw_optimizer.lr_warmup_steps": 700,
    "optimizer.adamw_optimizer.lr_decay_steps": 9000,
    "optimizer.adamw_optimizer.weight_decay": 0.0,
    "data_filter": filtering,
}
