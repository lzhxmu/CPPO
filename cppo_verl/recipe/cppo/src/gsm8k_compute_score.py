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

import re

def extract_last_number(text):
   """
   Extracts the last number appearing in the text.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The last number in the text, or None if no number is found.

   Explanation:
       1. Removes dollar signs and percent symbols from the text.
       2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
       3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
       4. Returns the found number as a float, or None if no match is found.
   """
   if text is None:
       return None
   text = text.replace('$', '').replace('%', '').replace(',', '')
   pattern = r'.*?(\d+\.?\d*)'
   matches = re.findall(pattern, text)
   return float(matches[-1]) if matches else None

def extract_answer_from_model_output(text):
   """
   Extracts the value from the last <answer> tag in the text.

   Args:
       text (str): The model-generated text containing XML-style <answer> tags.

   Returns:
       str or None: The content inside the <answer> tags, or None if no valid answer is found.

   Explanation:
       1. Splits the text on the <answer> tag to isolate content after the tag.
       2. Checks if at least one <answer> tag exists in the text.
       3. For the last <answer> segment:
          - Verifies it contains a closing </answer> tag.
          - Extracts only the content between the tags.
       4. Returns None if the answer is empty (just "...") or if tags are missing.
   """
   # Split on <answer> and take everything after the last occurrence
   parts = text.split("<answer>")
   if len(parts) < 2:  # No <answer> tag found
       return None
   last_part = parts[-1]

   # Extract content up to </answer>
   if "</answer>" not in last_part:
       return None
   answer = last_part.split("</answer>")[0].strip().replace(',', '')
   return None if answer == "..." else answer

def extract_single_number(text):
   """
   Extracts a single number from text if exactly one number is present.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.

   Explanation:
       1. Uses regex to find all numbers in the text (including negative numbers and decimals).
       2. If exactly one number is found, returns it as a float.
       3. If zero or multiple numbers are found, returns None.
   """
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None

def format_reward(solution_str):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer>$"
    match = re.match(pattern, solution_str, re.DOTALL | re.MULTILINE) 
    return 1.0 if match else 0.0 

def accuracy_reward(solution_str, ground_truth):
    """
    Assigns a reward based on the correctness of the model's answer.

    Explanation:
        1. Extracts the answer portion from each response using extract_answer_from_model_output.
        2. Assigns rewards based on matching criteria:
            - 2.0 points for an exact match
            - 1.5 points for numeric equivalence (when values match but format differs)
            - 0.0 points for incorrect answers
    """
    
    extracted = extract_answer_from_model_output(solution_str)
    if extracted == ground_truth:
        return 2.0, extracted
    else:
        ground_truth_num = extract_single_number(str(ground_truth))
        extracted_num = extract_single_number(str(extracted))
        
        if extracted_num is None:
            extracted_num = extract_last_number(str(solution_str))
            
        if ground_truth_num is not None and extracted_num is not None and ground_truth_num == extracted_num:
            return 1.5, extracted_num
        else:
            return 0, extracted_num if extracted_num is not None else -1


def compute_score(solution_str, ground_truth, **kwargs):
    R_format = format_reward(solution_str)
    R_accuracy, pred = accuracy_reward(solution_str, ground_truth)
    reward = R_format + R_accuracy
    acc = True if R_accuracy > 0 else False
    return {
        "score": reward,
        "acc": acc,
        # "pred": pred,
    }