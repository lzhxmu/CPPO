# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Math dataset to parquet format
"""

import re
import os

from verl.utils.hdfs_io import copy, makedirs
import argparse
from math_verify import LatexExtractionConfig, parse, verify


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    import datasets
    data_source = 'DigitalLearningGmbH/MATH-lighteval'
    dataset = datasets.load_dataset(data_source, 'default')
    train_dataset = dataset['train']

    test_data_source = 'HuggingFaceH4/MATH-500'
    dataset = datasets.load_dataset(test_data_source, 'default')
    test_dataset = dataset['test']

    instruction_following = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. And the answer should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>\n<answer> answer here </answer>."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('problem')

            question = question_raw

            answer_raw = example.pop('solution')
            solution = answer_raw
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": instruction_following},
                    {
                        "role": "user",
                        "content": question,}
                    ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
