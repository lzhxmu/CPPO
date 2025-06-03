# CPPO: verl version


This document presents the results of CPPO on the verl framework. Verl is a flexible, efficient, and production-ready RL training library for large language models (LLMs). Compared to the [OpenR1](https://github.com/huggingface/open-r1) framework, verl supports larger batch sizes under the same GPU memory constraints. Therefore, we use larger batch sizes in verl than in OpenR1. Due to differences in frameworks and training parameters, the accuracy and acceleration ratios of CPPO on verl may differ from those on OpenR1. With a suitable pruning rate, CPPO achieves significant acceleration without sacrificing accuracy compared to GRPO on verl. However, an excessively high pruning rate may remove high-quality completions and reduce training effectiveness, which is consistent with observations on the OpenR1 framework.




## 1. The results of CPPO on verl framework

### GSM8K
| Method                | Group Size (G) | Pruning Rate (P) | k  | Accuracy  | Training Time | Accelerate Ratio |
|-----------------------|---------------|------------------|----|-----------|---------------|------------------|
| Qwen2.5-1.5B-Instruct | -             | -                | -  |  55.42%  | -             | -                |
| GRPO                 | 16            | 0.00%            | 16 | 77.48%   | 8981.91s   | 1.00×            |
| CPPO                 | 16            | 50.00%           | 8  | 78.32%    | 4661.15s  | 1.93×            |
| CPPO                 | 16            | 75.00%           | 4  | 79.61%   | 2735.79s   | 3.28×            |
| CPPO                 | 16            | 87.50%           | 2  |78.70%    | 1932.56s   | 4.65×            |
| CPPO                 | 16            | 93.75%           | 1  | 76.65%    | 1206.53s  | 7.44×            |

Benefiting from the joint optimization of the verl framework and the CPPO algorithm, the training time for CPPO has been reduced to **1932.56s** (k=2) without sacrificing accuracy compared to GRPO. In contrast, under the OpenR1 framework, the training time for CPPO is **2813s** even with k=1.
| Method            | Group Size | Pruning Rate | k  | Accuracy | Time    | Accelerate Ratio | 
|------------------|------------|---------------|----|----------|---------|------------------|
| Qwen2.5-7B-Instruct | -          | -             | -  | 56.60%   | -       | -                | 
| GRPO             | 16         | 0.00%         | 16 | 77.00%   | 22191.40s  | 1.00×            | 
| CPPO             | 16         | 50.00%        | 8  | 77.20%   | 12652.02s  | 1.75×            |
| CPPO             | 16         | 75.00%        | 4  | 76.20%   | 7423.06s  | 2.99×            | 

Benefiting from the joint optimization of the verl framework and the CPPO algorithm, the training time for CPPO has been reduced to **12652s** (k=8) without sacrificing accuracy compared to GRPO. In contrast, under the OpenR1 framework, the training time for CPPO is **12959s** even with k=4.



## To Reproduce
### 1. Prepare the environment:
```bash
cd cppo_verl/
conda create -n verl python=3.10
conda activate verl
pip3 install torch==2.6.0
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
pip3 install math_verify
```
### 2. GSM8K:
You need two GPU with 80G memory to reproduce our results on GSM8K.
#### Training
##### GRPO
```bash
bash recipe/cppo/gsm8k_grpo.sh
```
##### CPPO
```bash
bash recipe/cppo/gsm8k_cppo.sh
```
#### Evaluation
```bash
bash recipe/cppo/gsm8k_eval.sh
```
### 3. Math:    
You need four GPU with 80G memory to reproduce our results on Math.
#### Training
##### GRPO
```bash
bash recipe/cppo/math_grpo.sh
```
##### CPPO
```bash
bash recipe/cppo/math_cppo.sh
```
#### Evaluation
```bash
bash recipe/cppo/math_eval.sh
```