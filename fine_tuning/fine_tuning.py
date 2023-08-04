import json
from dataclasses import asdict
from datetime import datetime

import torch
from src.modeling_longllama import LongLlamaForCausalLM
from transformers import HfArgumentParser, LlamaTokenizer, Trainer, TrainingArguments

from .arguments import DataArgs, ModelArgs, TokenizationArgs
from .data_processing import LOGGER, DataCollator, MixedTuneDataset
from .utils import get_packages, metrics_assign_group, non_numeric_to_str
import os

def main():
    hf_parser = HfArgumentParser((ModelArgs, DataArgs, TokenizationArgs, TrainingArguments))
    (
        model_args,
        data_args,
        tokenization_args,
        trainer_args,
    ) = hf_parser.parse_args_into_dataclasses()

    LOGGER.info(f"Preparing model {model_args.model_path}")
    model = LongLlamaForCausalLM.from_pretrained(
        model_args.model_path,
        mem_dtype=model_args.mem_dtype,
        last_context_length=model_args.last_context_length,
        torch_attention=model_args.torch_attention,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        gradient_checkpoint_every_ith=model_args.gradient_checkpoint_every_ith,
    )

    LOGGER.info(f"Preparing tokenizer {model_args.model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_path, padding_side="right", use_fast=False)

    LOGGER.info("Preparing dataset")
    dataset = MixedTuneDataset(data_args=data_args, tokenizer=tokenizer, tokenization_args=tokenization_args)

    LOGGER.info("Preparing trainer")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DataCollator(tokenizer=tokenizer),
    )

    model_args_dict = metrics_assign_group(asdict(model_args), "model_args")
    data_args_dict = metrics_assign_group(asdict(data_args), "data_args")
    tokenization_args_dict = metrics_assign_group(asdict(tokenization_args), "tokenization_args")
    trainer_args_dict = metrics_assign_group(asdict(trainer_args), "trainer_args")
    packages_dict = metrics_assign_group(get_packages(), "packages")
    all_params = {**model_args_dict, **data_args_dict, **tokenization_args_dict, **trainer_args_dict, **packages_dict}

    trainer.save_metrics("train", all_params, combined=True)

    str_params = json.dumps(all_params, indent=2)
    LOGGER.info(str_params)

    cur_time = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    with open(f"{trainer_args.output_dir}/params_{cur_time}.json", "w") as f:
        f.write(str_params)

    LOGGER.info("Running trainer")

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=os.path.join(trainer_args.output_dir, "final_model"))


if __name__ == "__main__":
    main()
