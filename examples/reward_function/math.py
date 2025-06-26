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
from typing import Any, Dict, List
import numpy as np

def get_answer(response: str):
    response = response.capitalize()
    result = []
    response = response.removeprefix('Answer:')
    response = response.split()
    for i in range(len(response)):
        result.append([int(j) for j in response[i]])
    return result

def fast_similarity_2d(list1, list2):
    a1 = np.asarray(list1)
    a2 = np.asarray(list2)
    return np.mean(a1 == a2)

def compute_shape(list1, list2):
    return 1.0 if np.asarray(list1).shape == np.asarray(list2).shape else 0.0

def compute_presence(list1, list2):
    set1 = {val for row in list1 for val in row}
    set2 = {val for row in list2 for val in row}
    return 1.0 if set1 == set2 else 0.0


def format_reward(response: str) -> float:
    pattern = re.compile(r'^Answer:\s*(\d+\s*)+$', re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0



def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        sim_score = 0.0
        try:
            resList = get_answer(response)
            ansList = get_answer(reward_input["ground_truth"])
            shape_score = compute_shape(resList, ansList)
            presence_score = compute_presence(resList, ansList)
            if shape_score == 1.0:
                sim_score = fast_similarity_2d(resList, ansList)
        except:
            shape_score = 0.0
            presence_score = 0.0
        scores.append(
            {
                "overall": format_score * 0.1 + sim_score * 0.6 + shape_score * 0.2 + presence_score * 0.1,
                "format": format_score,
                "similarity": sim_score,
                "shape": shape_score,
                "presence": presence_score,
            }
        )

    return scores
