import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm
from typing import Dict, List, Tuple

class ChatAlog():
    def __init__(self, input_data: pd.DataFrame) -> None:
        self.ORD = np.array(
            [[0.05, 0.80, 0.10, 0.05], # A
             [0.05, 0.25, 0.40, 0.30], # B
             [0.05, 0.05, 0.05, 0.85], # C
             [0.50, 0.40, 0.10, 0.00]]) # Z
        self.input_data = input_data
        self.conversation = None
        self.system_prompt = 'You are a helpful assistant.'

    def create_chat_history(self, system_prompt: str,
                            generate_times: int = 1000,
                            max_depth: int = 8,
                            init_weight: float = 0.05,
                            final_level_weight: float = 0.0000001,
                            current_punish: float = 0.01,
                            child_reward: float = 1000,
                            final_weighting_threshold: int = 2,
                            final_level_reward: float = 50):
        '''
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
        Return:
            self(ChatAlgo object)
        '''
        # Set system prompt
        if system_prompt:
            self.system_prompt = system_prompt
        # Clear conversation
        self.conversation = []
        # question weights matrix
        QM = self._create_question_matrix()
        A_level_counter = 0
        B_level_counter = 0
        C_level_counter = 0
        Z_level_counter = 0
        for times in tqdm(range(generate_times), desc='Generating Conversations'):
            # Initialize
            qa_list = []  # QA list
            rounds = 0  # reset round
            t_level = 0  # start from A level
            WM = np.copy(QM)  # Set weights matrix initialized as question matrix
            depth = random.randint(1, max_depth)  # Random sample the depth of QA rounds
            # generate chat
            while rounds < depth:
                # Get level weight
                LW = self._get_level_weights(rounds=rounds, init_weight = init_weight, final_level_weight = final_level_weight, last_level=t_level)
                # Weighting on sample matrix
                WM = self._level_weighting(WM, LW)
                
                # Verify whether initial round or not
                if rounds != 0:
                    # punish weighting
                    WM = self._punish_weighting(t, WM, punish=current_punish)  # weighting with punishment on former sampled cell
                    # child weighting
                    WM = self._child_weighting(t, WM, reward=child_reward)
                    # Z level weighting
                    if times > final_weighting_threshold:
                        WM = self._level_extra_weighting(level=3, weights=WM, reward=final_level_reward)
                
                # Sample from re-weighting
                t = self._sample_from_weight(WM)
                # next sample's level
                t_level = t[0]
                # next sample question
                qa_list.append(self.input_data[t[0]][t[1]][-2:].tolist())
                # Next round
                rounds += 1

                # Count number of each level's cell being sampled
                if t_level == 0:
                    A_level_counter += 1
                elif t_level == 1:
                    B_level_counter += 1
                elif t_level == 2:
                    C_level_counter += 1
                else:
                    Z_level_counter += 1
                    WM = np.copy(QM)

            self.conversation.append(self._formatter(qa_list))

        total_reach = sum([A_level_counter, B_level_counter, C_level_counter, Z_level_counter])
        print("Levels reach total: ", f"A: {round(A_level_counter/total_reach, 3)} | B: {round(B_level_counter/total_reach, 3)} | C: {round(C_level_counter/total_reach, 3)} | Z: {round(Z_level_counter/total_reach, 3)}")
        return self

    def _formatter(self, qalist: List[List[str]]) -> Dict:
        '''
        Turn simulated conversations into training data format
        '''
        return {
            "instruction": self.system_prompt,
            "input": qalist[-1][-2],
            "output": qalist[-1][-1],
            "history": qalist[:-1]
        }

    def to_json(self, output_path: str) -> None:
        '''
        Export simulated conversations to json file
        Args:
            output_path(str): The file path you wanna export
        '''
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation, f, ensure_ascii=False, indent=4)

    def _create_question_matrix(self) -> np.array:
        '''
        Turn input data into matrix(with corresponding position cell value = 1)
        '''
        question_matrix = []
        max_length = max([len(level) for level in self.input_data])
        for level in self.input_data:
            leng = len(level)
            question_matrix.append(([1]*leng + [0]*(max_length - leng)))
        return np.array(question_matrix)
    
    def _get_level_weights(self, rounds: int,
                           init_weight: float,
                           final_level_weight: float,
                           last_level: int) -> np.array:
        '''
        Get level weights matrix by last cell's level
        '''
        # initial round
        if rounds == 0:
            return np.diag([1-2*init_weight-final_level_weight, init_weight, init_weight, final_level_weight])
        # Other rounds
        else:
            return np.diag(self.ORD[last_level])
    
    def _level_weighting(self, QM:np.array, ILW: np.array) -> np.array:
        '''
        Weights each level with level weights matrix
        '''
        result = np.dot(ILW, QM)
        for level in result:
            leng = sum(1 for element in level if element != 0)
            if leng != 0:
                level /= leng
        return result

    def _sample_from_weight(self, weight_matrix: np.array) -> Tuple[int]:
        '''
        Get a sample from weights matrix
        '''
        # Define the shape of the weight matrix
        num_rows, num_cols = weight_matrix.shape
        # Flatten the weight matrix to use it for sampling
        flattened_weights = weight_matrix.flatten()
        # Sample an element location based on the provided weights
        sampled_index = np.random.choice(num_rows * num_cols, p=flattened_weights)
        # Convert the flattened index back to row and column indices
        row_index = sampled_index // num_cols
        col_index = sampled_index % num_cols
        return (row_index, col_index)

    def _punish_weighting(self, location: Tuple[int],
                          weights: np.array,
                          punish: float) -> np.array:
        '''
        Punish last selected cell's weight
        '''
        # Extract row and column indices from the location tuple
        i, j = location
        # Update weights
        updated_weights = np.copy(weights)
        updated_weights[i, j] *= punish
        # Normalize the updated weights
        updated_weights /= np.sum(updated_weights)
        return updated_weights

    def _child_weighting(self, location: Tuple[int],
                         weights: np.array,
                         reward: float) -> np.array:
        '''
        Reward childs of last selected cell's weights
        '''
        # Get parent UID
        parent_uid = self.input_data[location[0]][location[1]][0]
        # Find Child
        child_locations = self._find_child(parent_uid = parent_uid)
        if not child_locations:
            reward /= 10
            weights = self._level_extra_weighting(level = location[0], weights = weights, reward=reward)
            return weights
            # return self._level_extra_weighting(level = 3, weights = weights, reward=reward) # level = location[0]
        # Copy the weights to avoid modifying the original matrix
        updated_weights = np.copy(weights)
        # Increase weights for specified locations
        for child in child_locations:
            i, j = child
            updated_weights[i, j] *= (1 + reward)
        # Normalize the updated weights
        updated_weights /= np.sum(updated_weights)
        return updated_weights

    def _find_child(self, parent_uid: str) -> List[Tuple[int]]:
        '''
        Get childs position of last selected cell
        '''
        positions = []  # This will store the positions (row, column) of matching elements
        for row_index, row in enumerate(self.input_data):
            for col_index, (uid, parent, question, answer) in enumerate(row):
                if parent == parent_uid:
                    positions.append((row_index, col_index))
        return positions
    
    def _level_extra_weighting(self, level: int,
                               weights: np.array,
                               reward: float) -> np.array:
        '''
        Reward all cell in the level's weights
        '''
        updated_weights = np.copy(weights)
        updated_weights[level,] *= (1 + reward)
        # Normalize the updated weights
        updated_weights /= np.sum(updated_weights)
        return updated_weights