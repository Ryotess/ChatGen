import numpy as np
from typing import List, Tuple


def create_question_matrix(input_data: List[np.ndarray]) -> np.ndarray:
    """
    Turn input data into matrix(with corresponding position cell value = 1)
    """
    max_length = max(len(level) for level in input_data)
    question_matrix = [
        (np.ones(len(level)).tolist() + np.zeros(max_length - len(level)).tolist())
        for level in input_data
    ]
    return np.array(question_matrix)


def get_level_weights(
    rounds: int,
    init_weight: float,
    final_level_weight: float,
    last_level: int,
    ORD: np.ndarray,
) -> np.ndarray:
    """
    Get level weights matrix by last cell's level
    """
    if rounds == 0:
        return np.diag(
            [
                1 - 2 * init_weight - final_level_weight,
                init_weight,
                init_weight,
                final_level_weight,
            ]
        )
    return np.diag(ORD[last_level])


def level_weighting(QM: np.ndarray, ILW: np.ndarray) -> np.ndarray:
    """
    Weights each level with level weights matrix
    """
    result = np.dot(ILW, QM)
    for level in result:
        non_zero_count = np.count_nonzero(level)
        if non_zero_count:
            level /= non_zero_count
    return result


def sample_from_weight(weight_matrix: np.ndarray) -> Tuple[int, int]:
    """
    Get a sample from weights matrix
    """
    flattened_weights = weight_matrix.flatten()
    sampled_index = np.random.choice(flattened_weights.size, p=flattened_weights)
    return divmod(sampled_index, weight_matrix.shape[1])


def punish_weighting(
    location: Tuple[int, int], weights: np.ndarray, punish: float
) -> np.ndarray:
    """
    Punish last selected cell's weight
    """
    updated_weights = np.copy(weights)
    updated_weights[location] *= punish
    updated_weights /= np.sum(updated_weights)
    return updated_weights


def child_weighting(
    location: Tuple[int, int],
    weights: np.ndarray,
    reward: float,
    input_data: List[np.ndarray],
) -> np.ndarray:
    """
    Reward childs of last selected cell's weights
    """
    parent_uid = input_data[location[0]][location[1]][0]
    child_locations = find_child(parent_uid, input_data)

    updated_weights = np.copy(weights)
    if child_locations:
        for child in child_locations:
            updated_weights[child] *= 1 + reward
    else:
        reward /= 10
        updated_weights = level_extra_weighting(location[0], updated_weights, reward)
    updated_weights /= np.sum(updated_weights)
    return updated_weights


def find_child(parent_uid: str, input_data: List[np.ndarray]) -> List[Tuple[int, int]]:
    """
    Get childs position of last selected cell
    """
    return [
        (row_idx, col_idx)
        for row_idx, row in enumerate(input_data)
        for col_idx, (uid, parent, _, _) in enumerate(row)
        if parent == parent_uid
    ]


def level_extra_weighting(level: int, weights: np.ndarray, reward: float) -> np.ndarray:
    """
    Reward all cell in the level's weights
    """
    updated_weights = np.copy(weights)
    updated_weights[level] *= 1 + reward
    updated_weights /= np.sum(updated_weights)
    return updated_weights
