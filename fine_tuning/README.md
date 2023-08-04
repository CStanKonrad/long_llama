# LongLLaMA: Focused Transformer Training for Context Scaling

<div align="center">

<span style="padding: 1px;">[LongLLaMA-Instruct-3Bv1.1](https://huggingface.co/syzymon/long_llama_3b_instruct)</span> 

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CStanKonrad/long_llama/blob/main/long_llama_instruct_colab.ipynb)

</div>

This catalog contains code for instruction/chat tuning of the LongLLaMA models. Using this code we managed to tune the [LongLLaMA-3Bv1.1](https://huggingface.co/syzymon/long_llama_3b_v1_1) using one A100 80GB GPU in 44 hours. For tuning, we used [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) (instructions) and [zetavg/ShareGPT-Processed](https://huggingface.co/datasets/zetavg/ShareGPT-Processed) (chat) datasets. We call the created model [LongLLaMA-Instruct-3Bv1.1](https://huggingface.co/syzymon/long_llama_3b_instruct). We provide a colab demo of the model [here](https://colab.research.google.com/github/CStanKonrad/long_llama/blob/main/long_llama_instruct_colab.ipynb).
 
For more about LongLLaMA see the paper [Focused Transformer: Contrastive Training for Context Scaling](https://arxiv.org/abs/2307.03170).  


## Usage
Required packages are located in `requirements.txt`.  
Example configs are in files:
* [example_inst_ft_3b_low_budget.sh](example_inst_ft_3b_low_budget.sh) - only instruction tuning, smaller context
* [example_instchat_ft_3bv1.1_low_budget.sh](example_instchat_ft_3bv1.1_low_budget.sh) - instruction and chat tuning, config used for [LongLLaMA-Instruct-3Bv1.1](https://huggingface.co/syzymon/long_llama_3b_instruct)

To tune the model, simply run one of the scripts from the repo root directory. To manage the tuning process we use [Hugging Face trainer](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/trainer).  
For example, to create your own [LongLLaMA-Instruct-3Bv1.1](https://huggingface.co/syzymon/long_llama_3b_instruct) run `./fine_tuning/example_instchat_ft_3bv1.1_low_budget.sh`.

## Brief description of files
* [arguments.py](arguments.py) - see this file for the description of additional (non-Hugging Face) parameters
* [data_processing.py](data_processing.py) - used to process the data, this includes filtering, mixing chat and instruction data, padding etc.
* [fine_tuning.py](fine_tuning.py) - main script that runs the trainer
* [misc/trainer_state_of_LongLLaMA-Instruct-3v1.1.json](misc/trainer_state_of_LongLLaMA-Instruct-3v1.1.json) - tuning log for  [LongLLaMA-Instruct-3Bv1.1](https://huggingface.co/syzymon/long_llama_3b_instruct)


## Licensing
The code is available under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).  
Note that for finetunig we used [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) and [zetavg/ShareGPT-Processed](https://huggingface.co/datasets/zetavg/ShareGPT-Processed) datasets. Those datasets contain outputs of GPT models, what can affect the licensing of the models trained on them.

## Misc
Sometimes Hugging Face Trainer can pick the logger by default. If you run into problems with this you can manually set the logger for example by adding `--report_to "tensorboard"` inside the script.  

If you plan to use this codebase for different models, please note how the padding is applied. Note also that attention is masked for padding tokens. 

