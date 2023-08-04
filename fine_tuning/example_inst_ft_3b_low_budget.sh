#!/bin/bash
EXP_NAME="example_inst_ft_3b_low_budget"
python3 -m fine_tuning.fine_tuning \
    --run_name "$EXP_NAME" \
    --output_dir "$EXP_NAME"/ \
    --model_path "syzymon/long_llama_3b" \
    --torch_dtype float32 \
    --data_type "instructions" \
    --data_path "Open-Orca/OpenOrca" \
    --data_revision "f0823c7ffc48c9d33a42f16cf0b885fed4a7d0a1" \
    --dataset_split "train" \
    --prompt_field "system_prompt" \
    --post_prompt_text "
" \
    --question_field "question" \
    --post_question_text "
" \
    --response_field "response" \
    --last_context_length 768 \
    --max_input_length 1024 \
    --max_output_length 1024 \
    --max_total_length 1024 \
    --always_pad True \
    --random_pad True \
    --max_steps 4500 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1.0e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
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
