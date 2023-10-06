# LongLLaMA-Code 7B Instruct


<div align="center">

<table>
  <tr>
    <th style="font-size: 120%"> >_ 🎓 <a href="https://huggingface.co/syzymon/long_llama_code_7b_instruct">LongLLaMA-Code 7B Instruct</a> 📑🗨 </th>
  </tr>
  <tr>
    <td align="center">
    <a  href="https://colab.research.google.com/github/CStanKonrad/long_llama/blob/main/long_llama_code_instruct_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
    </td>
    
 </tr>
</table>

</div>


## TLDR
[LongLLaMA-Code 7B Instruct](https://huggingface.co/syzymon/long_llama_code_7b_instruct) is [LongLLaMA-Code 7B](https://huggingface.co/syzymon/long_llama_code_7b) tuned on [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct), [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), and [ShareGPT-Processed](https://huggingface.co/datasets/zetavg/ShareGPT-Processed) datasets. It can answer basic questions about research papers and code. It can also perform a simple code refactoring. You can try the quantized version of the model using a free GPU in [Google Colab](https://colab.research.google.com/github/CStanKonrad/long_llama/blob/main/long_llama_code_instruct_colab.ipynb).

## Tuning

### Code
The model was tuned on a TPU v3-128 pod with 128 batch size.  
For tuning, we have used the data preparation pipeline available in [instruction_fine_tuning](.).
However, we have replaced the Hugging Face Trainer with a modification of [FoT continued pretraining code](../fot_continued_pretraining). This modification boils down to propagating the memory cache throughout the model (basically reproducing the Pytorch inference code functionality in JAX).

### Training
Here, we present the basic information about how the model was tuned. For more details, see [misc/LongLLaMA_Code_7B_Instruct_details.py](misc/LongLLaMA_Code_7B_Instruct_details.py).


All inputs were truncated and randomly padded (left/right) to 3072 tokens.  
The last context length was set to 1536.  
The model was trained for 9k steps, started with a learning rate of 1.2e-5, 700 steps of warmup, and finished with a learning rate of 0.  
The optimizer was adamw.  

The question prompt (`pre_question_text`) was:
```
You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can.\n\n
```

To trigger the model answer one can use:
```
\nAnswer: 
```

The chat prompt was:
```
A chat between a user (denoted as USER:) and an artificial intelligence assistant (denoted as ASSISTANT:). The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n
```

To denote the assistant one can write:
```
\nASSISTANT: 
```

To denote the user one can write:
```
\nUSER: 
```

### Datasets and sampling probability
* 0.71 - [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
* 0.16, - [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) questions with less than 5k chars
* 0.08, - [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) questions above 5k chars but below 12k chars
* 0.02 - [zetavg/ShareGPT-Processed](https://huggingface.co/datasets/zetavg/ShareGPT-Processed) conversations below 6k chars
* 0.01 - [zetavg/ShareGPT-Processed](https://huggingface.co/datasets/zetavg/ShareGPT-Processed) conversations above 6k chars but below 12k chars

To improve the quality of the data, the datasets were filtered using regular expressions.  



## License
The instruction/chat-tuned models are for research purposes only.
[LongLLaMA-Code 7B Instruct](https://huggingface.co/syzymon/long_llama_code_7b_instruct) is [LongLLaMA-Code 7B](https://huggingface.co/syzymon/long_llama_code_7b) tuned on [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct), [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), and [ShareGPT-Processed](https://huggingface.co/datasets/zetavg/ShareGPT-Processed) datasets. Note that those datasets contain outputs from ChatGPT. See also the [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) license.

## Acknowledgements
We gratefully acknowledge the TPU Research Cloud program, which was instrumental to our research by providing significant computational resources.

