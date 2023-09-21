# LongLLaMA: Focused Transformer Training for Context Scaling - FoT fine-tuning


This catalog contains the FoT fine-tuning code for LongLLaMA.  
It is based on a subset of [EasyML](https://github.com/young-geng/EasyLM) and introduces small changes that allow us to fine-tune LLaMA models with FoT.  
See [Brief description of files and changes](#Brief-description-of-files-and-changes) for additional details.  
For details about LongLLaMA and FoT see:
* **[FoT brief explanation](FoT/README.md)**
* [Focused Transformer: Contrastive Training for Context Scaling](https://arxiv.org/abs/2307.03170).

## Usage
Required packages are located in [requirements.txt](requirements.txt) (more specific versions in [requirements.freeze.txt](requirements.freeze.txt)).  
Before installing requirements change the JAX version to the one that is appropriate for your accelerator.

To run an experiment simply place yourself in this catalog and type 

`python3 -m running_utils.runner --config configs/setup_test.json`


## Brief description of files and changes

To understand how FoT is implemented, it should suffice to look at files in the [FoT](FoT/) subdirectory and 
`FlaxLLaMAAttention` in [EasyLM/models/llama/llama_model.py](EasyLM/models/llama/llama_model.py).

The **main files** with **the method** and **the data pipeline** are:
* [FoT/cross_batch.py](FoT/cross_batch.py) - contains the implementation of cross-batch that uses positional encodings for local context and encodes other elements as if they were at the position of the first element in the local context. 
* [FoT/data_pipeline.py](FoT/data_pipeline.py) - contains the implementation of a doc-aware data pipeline, which is a data pipeline that assigns docs to elements of the batch.

Changed files:
* [EasyLM/data.py](EasyLM/data.py) - was slightly modified to make use of the data pipelines mentioned above
* [EasyLM/models/llama/llama_model.py](EasyLM/models/llama/llama_model.py) - example cross-batch configs were added and cross-batch was inserted into the `FlaxLLaMAAttention` module.
* [EasyLM/models/llama/llama_train.py](EasyLM/models/llama/llama_train.py) - was modified to support accumulated train logs, custom configuration of gradient accumulation, and separate model configuration for eval. WandB was replaced.

Added utility files include:
* [EasyLM/training_utils.py](EasyLM/training_utils.py) - utilities for training
* [EasyLM/logging_utils.py](EasyLM/logging_utils.py) - utilities for log accumulation, etc.
* [running_utils/runner_utils.py](running_utils/runner_utils.py) - utilities for running the training from a predefined config
* [running_utils/runner.py](running_utils/runner.py) - allows to run the training using configuration saved in a JSON file

Tokenizers adopted from [OpenLLaMA](https://github.com/openlm-research/open_llama) are in the [tokenizers](tokenizers/) directory.  
Examples are located in the [configs](configs/) directory.

Some files were removed as they are not necessary to create LongLLaMA.



## Troubleshooting

* In case the experiment is getting out of memory error, try to change:
  + sharding parameters (`mesh_dim` in config)
  + gradient checkpointing (see [EasyLM/models/llama/llama_model.py](EasyLM/models/llama/llama_model.py))
  + enable scan attention for cross batch (`scan_cross_batch` parameter)
* If code freezes on a TPU pod it may mean that the workers are not executing the same code, or some worker has failed to start
* If you get low results make sure that `rope_theta` in Rotary Positional Encodings (see [EasyLM/models/llama/llama_model.py](EasyLM/models/llama/llama_model.py)) matches the one of the base model (note the difference between LLaMA and Code Llama)


## Misc
### Running on TPU pods
To run the code on a TPU pod you should run the same code on each worker node with `pod_job` being set to `True`.  
This can be achieved by running  

`python3 -m running_utils.runner --config configs/chosen_config.json --pod_job`  

on each of the worker nodes. 

### Additional functionalities
The code adopted from [EasyML](https://github.com/young-geng/EasyLM) may have additional functionalities that were not tested with the provided implementation of FoT.
In particular, we have not tested how the implementation scales on multiple GPUs. 

## Credits
Below we attach the credis from the [EasyML](https://github.com/young-geng/EasyLM) repository:
* The LLaMA implementation is from [JAX_llama](https://github.com/Sea-Snell/JAX_llama)
* Most of the JAX utilities are from [mlxu](https://github.com/young-geng/mlxu)
* The [EasyML](https://github.com/young-geng/EasyLM) codebase is heavily inspired by [JAXSeq](https://github.com/Sea-Snell/JAXSeq)
