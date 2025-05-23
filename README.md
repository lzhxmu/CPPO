# CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models


<p align="center">
<a href="https://arxiv.org/abs/2503.22342"> <img src='https://img.shields.io/badge/arXiv-2503.22342-b31b1b.svg'></a> 
</p>

## Abstract
This paper introduces Completion Pruning Policy Optimization (CPPO) to accelerate the training of reasoning models based on Group Relative Policy Optimization (GRPO). GRPO, while effective, incurs high training costs due to the need for sampling multiple completions for each question. Our experiment and theoretical analysis reveals that the number of completions impacts model accuracy yet increases training time multiplicatively, and **not all completions contribute equally to policy training**---t**heir contribution depends on their relative advantage**. To address these issues, we propose CPPO, which prunes completions with low absolute advantages, significantly reducing the number needed for gradient calculation and updates. Additionally, we introduce a dynamic completion allocation strategy to maximize GPU utilization by incorporating additional questions, further enhancing training efficiency. Experimental results demonstrate that CPPO achieves up to **$8.32\times$** speedup on GSM8K and **$3.51\times$** on Math while preserving or even enhancing the accuracy compared to the original GRPO.


## Main Results

### GSM8K
| Method                | Group Size (G) | Pruning Rate (P) | k  | Accuracy  | Training Time | Accelerate Ratio |
|-----------------------|---------------|------------------|----|-----------|---------------|------------------|
| Qwen2.5-1.5B-Instruct | -             | -                | -  | 55.72%    | -             | -                |
| GRPO                 | 16            | 0.00%            | 16 | 77.05%    | 23393s        | 1.00×            |
| CPPO                 | 16            | 50.00%           | 8  | 77.67%    | 12930s        | 1.81×            |
| CPPO                 | 16            | 75.00%           | 4  | 78.81%    | 7159s         | 3.27×            |
| CPPO                 | 16            | 87.50%           | 2  | 80.41%    | 4781s         | 4.89×            |
| CPPO                 | 16            | 93.75%           | 1  | 78.20%    | 2813s         | 8.32×            |

### Math & Out-of-Distribution tasks  

| Method            | Group Size | Pruning Rate | k  | Accuracy | Time    | Accelerate Ratio | AMC 2023 | AIME 2024 |
|------------------|------------|---------------|----|----------|---------|------------------|----------|-----------|
| Qwen2.5-7B-Instruct | -          | -             | -  | 55.20%   | -       | -                | 25.62%   | 5.00%     |
| GRPO             | 16         | 0.00%         | 16 | 75.20%   | 33902s  | 1.00×            | 46.88%   | 5.83%     |
| CPPO             | 16         | 50.00%        | 8  | 75.20%   | 20550s  | 1.65×            | 53.12%   | 10.00%    |
| CPPO             | 16         | 75.00%        | 4  | 77.20%   | 12959s  | 2.62×            | 49.38%   | 6.67%     |
| CPPO             | 16         | 87.50%        | 2  | 75.20%   | 9657s   | 3.51×            | 46.25%   | 8.33%     |
| CPPO             | 16         | 93.75%        | 1  | 72.80%   | 8375s   | 4.05×            | 45.00%   | 5.83%     |


## To Reproduce

### 1. Prepare the environment:
```bash
conda create -n cppo python=3.11
conda activate cppo
pip install vllm==0.7.2
pip install setuptools
pip install flash-attn --no-build-isolation
pip install -e ".[dev]"
```

### 2. GSM8K:

#### Training
You need **two** GPU with 80G memory to reproduce our results on GSM8K.
##### GRPO
```bash
sh scripts/GRPO_gsm.sh
```
##### CPPO
```bash
sh scripts/CPPO_gsm.sh
```



#### Evaluation


##### Qwen2.5-1.5B-Instruct
```bash
sh scripts/Eval_qwen2.5-1.5b.sh
```
##### CPPO-1.5B-n-16-0.875
```bash
sh scripts/Eval_gsm.sh
```
You can download the ckpt from <a href="https://huggingface.co/Stardust1956/CPPO-1.5b-n-16-0.875">huggingface 🤗</a>.

### 4. Math:

#### Training
You need **four** GPU with 80G memory to reproduce our results on Math.
##### GRPO
```bash
sh scripts/GRPO_math.sh
```
##### CPPO
```bash
sh scripts/CPPO_math.sh
```

#### Evaluation
##### Qwen2.5-7B-Instruct
```bash
sh scripts/Eval_qwen2.5-7b.sh
```
##### CPPO-7B-n-16-0.75
```bash
sh scripts/Eval_math.sh
```
You can download the ckpt from <a href="https://huggingface.co/Stardust1956/CPPO-7b-n-16-0.75">huggingface 🤗</a>.


## Acknowledgments
We are very grateful to the [Open R1](https://github.com/huggingface/open-r1) teams for creating awesome repo.



## Citation
```bibtex
@article{lin2025cppo,
  title={CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models},
  author={Lin, Zhihang and Lin, Mingbao and Xie, Yuan and Ji, Rongrong},
  journal={arXiv preprint arXiv:2503.22342},
  year={2025}
}
```
