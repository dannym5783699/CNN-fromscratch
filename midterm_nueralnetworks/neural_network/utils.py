import numpy as np

def get_batches(X : np.ndarray, Y : np.ndarray, batch_size : int):

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    if batch_size is None:
        yield X, Y
    else:
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], Y[batch_indices]