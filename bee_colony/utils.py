import numpy as np


def get_random_other_index(current_index, num_indices):
    indices = list(range(num_indices))
    random_index = current_index
    while random_index == current_index:
        random_index = np.random.randint(0, num_indices)
    return random_index


def assign_probabilities(probabilities, ordered_indices):
    if len(probabilities) != len(ordered_indices):
        raise ValueError(f'Length mismatch: {len(probabilities)} != {len(ordered_indices)}')

    sorted_probabilities = np.zeros((len(probabilities),))
    for i, idx in enumerate(ordered_indices):
        sorted_probabilities[idx] = probabilities[i]
    return sorted_probabilities


def uniform_init(bounds):
    """
    Uniform initialization function.
    """
    if not isinstance(bounds, (np.ndarray, list)):
        raise ValueError('Bounds must be a list or numpy array')
    elif np.ndim(bounds) != 2:
        ndim = np.ndim(bounds)
        raise ValueError(f'Bounds must be a 2D array but got an array of dim {ndim}')

    dimensions = len(bounds)
    min_bounds = np.array([b[0] for b in bounds])
    max_bounds = np.array([b[1] for b in bounds])

    def init_fn(population_size):
        return np.random.uniform(min_bounds, max_bounds, size=(population_size, dimensions))

    return init_fn
