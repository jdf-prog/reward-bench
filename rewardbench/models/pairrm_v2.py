# Copyright 2023 AllenAI. All rights reserved.
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

# copied partially from https://github.com/yuchenlin/LLM-Blender/blob/main/llm_blender/pair_ranker/pairrm.py
# and added pairwise tokenization function from https://huggingface.co/llm-blender/PairRM-hf
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from string import Template


input_template = """\
Given the following conversation histories and responses from two assistants, determine the better assistant in response to the user.
User responses start with "USER: ", and assistant responses start with "ASSISTANT_A: " or "ASSISTANT_B: ". 
If two assistants give the same response, response from assistant B will be denoted as "<SAME_AS_A>".

$history

If you think Assistant A is better, output "A". If you think Assistant B is better, output "B". If you think they are equally good, output "C".
Your Choice: [A/B/C]
"""

def format_input(
    prompts: List[str],
    responses_A: List[str],
    responses_B: List[str],
):
    """
    Format input data for PairRM.
    item:
    ```
    Given the following conversation histories and responses from two assistants, determine the better assistant in response to the user. 
    User responses start with "USER: ", and assistant responses start with "ASSISTANT_A: " or "ASSISTANT_B: ". 
    ### Turn 1
    USER: ...
    ASSISTANT_A: ...
    ASSISTANT_B: ...
    ### Turn 2:
    USER: ...
    ASSISTANT_A: ...
    ASSISTANT_B: ...
    ...
    If you think Assistant A is better, output "A". If you think Assistant B is better, output "B". If you think they are equally good, output "C".
    Your Choice: [A/B/C]
    ```
    """

    history = ""
    for i, (prompt, response_A, response_B) in enumerate(zip(prompts, responses_A, responses_B)):
        history += f"### Turn {i+1}\n"
        history += f"USER: {prompt}\n"
        history += f"ASSISTANT_A: {response_A}\n"
        if response_B == response_A:
            history += f"ASSISTANT_B: <SAME_AS_A>\n"
        else:
            history += f"ASSISTANT_B: {response_B}\n"
        history += "\n"
    return Template(input_template).substitute(history=history)

def tokenize_conv_pair(tokenizer, convAs: List[str], convBs: List[str], **kwargs):
    """Compare two conversations by takeing USER turns as inputs and ASSISTANT turns as candidates
        Multi-turn conversations comparison is also supportted.
        a conversation format is:
        ```python
        [
            {
                "content": "hello",
                "role": "user"
            },
            {
                "content": "hi",
                "role": "assisstant"
            },
            ...
        ]
        ```
    Args:
        convAs (List[List[dict]]): List of conversations
        convAs (List[List[dict]]): List of conversations
    """

    for c in convAs + convBs:
        if not all([c[i]["role"] == "assistant" for i in range(1, len(c), 2)]):
            print(c)

        assert len(c) % 2 == 0, "Each conversation must have even number of turns"
        assert all([c[i]["role"] == "user" for i in range(0, len(c), 2)]), "Each even turn must be USER"
        assert all([c[i]["role"] == "assistant" for i in range(1, len(c), 2)]), "Each odd turn must be ASSISTANT"
    # check conversations correctness
    assert len(convAs) == len(convBs), "Number of conversations must be the same"
    for c_a, c_b in zip(convAs, convBs):
        assert len(c_a) == len(c_b), "Number of turns in each conversation must be the same"
        assert all(
            [c_a[i]["content"] == c_b[i]["content"] for i in range(0, len(c_a), 2)]
        ), "USER turns must be the same"

    all_messages = []
    for c_a, c_b in zip(convAs, convBs):
        queries = [c_a[i]["content"] for i in range(0, len(c_a), 2)]
        responses_A = [c_a[i]["content"] for i in range(1, len(c_a), 2)]
        responses_B = [c_b[i]["content"] for i in range(1, len(c_b), 2)]
        prompt = format_input(queries, responses_A, responses_B)

        all_messages.append([
            {"role": "user", "content": prompt}
        ])
    
    input_ids = tokenizer.apply_chat_template(all_messages, add_generation_prompt=True, return_tensors="pt", **kwargs)
    return input_ids

class PairRMV2Pipeline:
    """
    This class outputs a delta rather than a score for each.
    """

    def __init__(self, task, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side='left'
        # turn off gradients for model and set in eval mode
        self.model.eval().requires_grad_(False)

    def __call__(self, candidates_A: List[str], candidates_B: List[str], output_logits=False, **kwargs):
        
        input_ids = tokenize_conv_pair(self.tokenizer, candidates_A, candidates_B, **kwargs).to(self.model.device)

        generation_config = {
            "max_new_tokens": 20,
            "do_sample": False,
        }
        output_ids = self.model.generate(input_ids=input_ids, **generation_config)
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        logits = []
        for i, output_text in enumerate(output_texts):
            if output_text=="A":
                logits.append(1)
            elif output_text=="B":
                logits.append(-1)
            else:
                logits.append(0)
        return torch.tensor(logits)