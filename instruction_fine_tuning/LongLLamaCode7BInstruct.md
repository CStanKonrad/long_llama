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
However, we have replaced the Hugging Face Trainer with a modification of [FoT continued pretraining code](../fot_continued_pretraining). This modification boils down to propagating the memory cache throughout the model (basically reproducing the Pytorch code functionality in JAX).

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


## GSM8K Eval

<div align="center">

<table>
  <tr>
    <th style="font-size: 120%"> PoT</th>
    <th style="font-size: 120%"> CoT</th>
  </tr>
  <tr>
    <td align="center">
    64.5%
    </td>
    <td align="center">
    42.2%
    </td>
    
 </tr>
</table>

</div>



For the PoT evaluation, we have used the code from [TIGER-AI-Lab/MAmmoTH](https://github.com/TIGER-AI-Lab/MAmmoTH/tree/main/math_eval).
As our model was trained with a different prompt, we have added the following code to [prompt_utils.py](https://github.com/TIGER-AI-Lab/MAmmoTH/blob/7f24220b8e6f50aae200096449571a6246571f9f/math_eval/prompt_utils.py)
```python3
def get_fot_prompt_v1(qas: list):
    trigger = "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can.\n\n"

    prompt = ""
    for q, a in qas:
        prompt += trigger + q + "\nAnswer: " + a +"\n\n"
    
    prefix = trigger + "{query}\nAnswer: "

    return prompt, prefix
```
Note that [run_open.py](https://github.com/TIGER-AI-Lab/MAmmoTH/blob/7f24220b8e6f50aae200096449571a6246571f9f/math_eval/run_open.py) modifies the question for PoT (by appending `Let's write a program`) so that the example input looks like this:
```
You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can.

A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take? Let's write a program.
Answer: 
```
and the example output looks like this:
```
# define the variables
blue_fiber = 2
white_fiber = blue_fiber / 2

# calculate the total fiber
total_fiber = blue_fiber + white_fiber

# print the result
print(total_fiber)
```
The run parameters (after adding the FoT prompt) are as follows:
```
--batch_size 8 \
--dataset gsm8k \
--dtype bfloat16 \
--form fot_custom1 \
--model "syzymon/long_llama_code_7b_instruct" \
--model_max_length 1500 \
--shots 0 \
--stem_flan_type pot_prompt
```


For 8-shot CoT, we use the code from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and additionally add the `<bos>` token.
We take the examples from [Chain-of-Thought Prompting Elicits Reasoning
in Large Language Models](https://browse.arxiv.org/pdf/2201.11903.pdf). 
To be more precise, the examples are:
```
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

```
As the answer we extract the last thing that matches:
`(?!\,)(?!\.)(\-?[0-9\.\,]+)`

## License
The instruction/chat-tuned models are for research purposes only.
[LongLLaMA-Code 7B Instruct](https://huggingface.co/syzymon/long_llama_code_7b_instruct) is [LongLLaMA-Code 7B](https://huggingface.co/syzymon/long_llama_code_7b) tuned on [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct), [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), and [ShareGPT-Processed](https://huggingface.co/datasets/zetavg/ShareGPT-Processed) datasets. Note that those datasets contain outputs from ChatGPT. See also the [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) license.

## Acknowledgements
We gratefully acknowledge the TPU Research Cloud program, which was instrumental to our research by providing significant computational resources.

