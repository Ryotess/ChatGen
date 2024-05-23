import numpy as np
import random
import json
from tqdm import tqdm
from typing import List, Tuple, Dict
from chatgen.weight_utils import (
    create_question_matrix,
    get_level_weights,
    level_weighting,
    sample_from_weight,
    punish_weighting,
    child_weighting,
    level_extra_weighting,
    find_child,
)


class ChatAlgo:
    def __init__(self, input_data: List[np.ndarray]) -> None:
        self.ORD = np.array(
            [
                [0.05, 0.80, 0.10, 0.05],  # A
                [0.05, 0.25, 0.40, 0.30],  # B
                [0.05, 0.05, 0.05, 0.85],  # C
                [0.50, 0.40, 0.10, 0.00],  # Z
            ]
        )
        self.input_data = input_data
        self.conversation = None
        self.system_prompt = "You are a helpful assistant."

    def create_chat_history(
        self,
        system_prompt: str,
        generate_times: int = 1000,
        max_depth: int = 8,
        init_weight: float = 0.05,
        final_level_weight: float = 0.0000001,
        current_punish: float = 0.01,
        child_reward: float = 1000,
        final_weighting_threshold: int = 2,
        final_level_reward: float = 50,
    ) -> None:
        """
        Simulate a set of chat history from input data
        Args:
            generate_times(int): The number of conversations to be simulated
            max_depth(int): The max depth of conversation you wanna generate
            init_weight(float): Initial weighting for levels(lower value will make higher probability of First level to be sampled)
            final_level_weight(float): Level Z initial weight
            current_punish(float): The punish weighting to be multiplied on last selected question
            child_reward(float): The reward weighting to be multiplied on all the child questions of last selected question
            final_weighting_threshold(int): The number of rounds that after this round the final level question's weights should increased
            final_level_reward(float): The reward weighting of final level question after
        """
        if system_prompt:
            self.system_prompt = system_prompt

        self.conversation = []
        QM = create_question_matrix(self.input_data)
        level_counters = [0, 0, 0, 0]

        for _ in tqdm(range(generate_times), desc="Generating Conversations"):
            qa_list, rounds, t_level = [], 0, 0
            WM = np.copy(QM)
            depth = random.randint(1, max_depth)

            while rounds < depth:
                LW = get_level_weights(
                    rounds, init_weight, final_level_weight, t_level, self.ORD
                )
                WM = level_weighting(WM, LW)

                if rounds != 0:
                    WM = punish_weighting(t, WM, current_punish)
                    WM = child_weighting(t, WM, child_reward, self.input_data)
                    if rounds > final_weighting_threshold:
                        WM = level_extra_weighting(3, WM, final_level_reward)

                t = sample_from_weight(WM)
                t_level = t[0]
                qa_list.append(self.input_data[t[0]][t[1]][-2:].tolist())
                rounds += 1

                level_counters[t_level] += 1

                if t_level == 3:
                    WM = np.copy(QM)

            self.conversation.append(self._formatter(qa_list))

        total_reach = sum(level_counters)
        print(
            "Levels reach total: ",
            " | ".join(
                f"{level}: {round(count/total_reach, 3)}"
                for level, count in zip("ABCZ", level_counters)
            ),
        )

    def _formatter(self, qalist: List[List[str]]) -> Dict:
        """
        Turn simulated conversations into training data format
        """
        return {
            "instruction": self.system_prompt,
            "input": qalist[-1][0],
            "output": qalist[-1][1],
            "history": qalist[:-1],
        }

    def to_json(self, output_path: str) -> None:
        """
        Export simulated conversations to json file
        Args:
            output_path(str): The file path you wanna export
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation, f, ensure_ascii=False, indent=4)

    def sample_output(self):
        sample = random.choice(self.conversation)
        rounds = 1

        print(f"Instructions: {sample['instruction']}", end="\n---------\n")

        for question, answer in sample["history"]:
            print(f"Q: {question}\nA: {answer}")
            rounds += 1
        print(f"Q: {sample['input']}\nA: {sample['output']}", end="\n---------\n")

        print("Total Rounds: ", rounds)
