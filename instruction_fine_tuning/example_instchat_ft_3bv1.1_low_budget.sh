#!/bin/bash
EXP_NAME="example_instchat_ft_3bv1.1_low_budget"
python3 -m instruction_fine_tuning.fine_tuning \
    --run_name "$EXP_NAME" \
    --output_dir "$EXP_NAME"/ \
    --model_path "syzymon/long_llama_3b_v1_1" \
    --torch_dtype float32 \
    --data_type "instructions,chat" \
    --data_path "Open-Orca/OpenOrca,zetavg/ShareGPT-Processed" \
    --data_revision "f0823c7ffc48c9d33a42f16cf0b885fed4a7d0a1,15968d6dfa02529988da12e382af3ab7c857ebcd" \
    --data_filter "<,>lang<M>^en$<;>conversations.<int>*^.value<M>(?i)^((?!openai).)*$<;>conversations.<int>*^.value<M>^((?!DAN).)*$<;>conversations.<int>0.value<LENLT>8000" \
    --dataset_split "train,train" \
    --data_proportions 0.99 0.01 \
    --prompt_field "system_prompt" \
    --post_prompt_text "
" \
    --question_field "question" \
    --post_question_text "
" \
    --response_field "response" \
    --chat_conversations_field "conversations" \
    --chat_data_field "markdown" \
    --chat_source_name_field "from" \
    --chat_model_source_name "gpt" \
    --chat_initial_prompt "A chat between a user (denoted as USER:) and an artificial intelligence assistant (denoted as ASSISTANT:). The assistant gives helpful, detailed, and polite answers to the user's questions.

" \
    --chat_model_response_prefix "
ASSISTANT: " \
    --chat_human_response_prefix "
USER: " \
    --last_context_length 1024 \
    --max_input_length 2048 \
    --max_output_length 2048 \
    --max_total_length 2048 \
    --always_pad True \
    --random_pad True \
    --max_steps 4000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_steps 200 \
    --lr_scheduler_type "cosine" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --save_steps 500 \
    --logging_strategy "steps" \
    --logging_steps 25 \
    --gradient_checkpointing True \
    --tf32 True \
    --bf16 True
