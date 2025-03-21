import numpy as np
import matplotlib.pyplot as plt
 
    
    
def get_top_value_random_lastelement(vec, topk):
    """
    Selects the top-k highest values from a vector, but randomly samples from tied values at the cutoff.

    This avoids bias when many values are tied for the last place in the top-k.

    Parameters:
    - vec: 1D array of values
    - topk: number of values to select

    Returns:
    - s_idx: array of selected indices
    """
    
    sort_idx = np.argsort(vec)[::-1]
    top_vec = sort_idx[:topk]
    val = vec[sort_idx[-1]]
    last_idx = np.where(vec==val)[0]
    while vec[top_vec[-1]] == val:
        top_vec = top_vec[:-1]
        if len(top_vec) == 0: break
    leftover = topk - len(top_vec)
    leftover_idx = np.random.choice(len(last_idx), leftover, replace=False)
    s_idx = np.hstack([top_vec, last_idx[leftover_idx]])
    
    return s_idx
                
                
                

def max_value_indicator(matrix):
    """
    Convert the index of the maximum element in each row to one-hot encoding.

    Args:
    - matrix: 2D array (matrix)

    Returns:
    - one_hot_matrix: 2D array (matrix) with one-hot encoding of maximum element indices in each row
    """
    max_indices = np.argmax(matrix, axis=1)
    num_cols = matrix.shape[1]
    one_hot_matrix = np.zeros_like(matrix)
    rows = np.arange(matrix.shape[0])
    one_hot_matrix[rows, max_indices] = 1
    return one_hot_matrix

def random_indicator_matrix(n, k):
    """
    Creates a binary indicator matrix where each row has exactly one randomly selected 1.

    Parameters:
    - n: number of rows
    - k: number of columns

    Returns:
    - indicator_matrix: binary matrix of shape (n, k) with one active element per row
    """
    
    indicator_matrix = np.zeros((n, k), dtype=int)
    for i in range(n):
        indices = np.random.choice(k, 1, replace=False)
        indicator_matrix[i, indices] = 1
    return indicator_matrix


def one_hot_encode(labels, num_classes):
    """
    Convert a list of labels to their corresponding one-hot encodings.

    Args:
    - labels: List of labels (integers)
    - num_classes: Total number of classes

    Returns:
    - one_hot_matrix: NumPy array containing the one-hot encodings
    """
    one_hot_matrix = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_matrix[i, label] = 1
    return one_hot_matrix


def block_diagonal(matrices):
    """
    Constructs a block diagonal matrix from a list of smaller 2D arrays.

    Each input matrix is placed along the diagonal of a larger matrix, with zeros elsewhere.

    Parameters:
    - matrices: list of 2D NumPy arrays

    Returns:
    - block_matrix: large block-diagonal NumPy array
    """
    num_rows = np.sum([m.shape[0] for m in matrices])
    num_cols = np.sum([m.shape[1] for m in matrices])
    
    block_matrix = np.zeros((num_rows,num_cols))
    
    offsetrow = 0
    offsetcol = 0
    
    for m in matrices:
        r,c = m.shape
        
        block_matrix[offsetrow:(offsetrow+r), offsetcol:(offsetcol+c)] = m
        offsetrow += r
        offsetcol += c
    return block_matrix






def waterfill(r):
    """

    Solves a constrained min-max problem using a waterfilling algorithm.
    
    min_{q\in simplex} max_{y\in {1,...,K}} E_{ypred ~ q}[1_{y\neq ypred}] + R(y)
    
    By writing r such that r_k = R(e_k), we can rewrite this problem as

    min_{q\in simplex} max_{k \in {1,...,K}} (1-q+r)_k
    
    This can be done through waterfilling.

    1. Start with q = 0
    2. At each iteration, find k = argmax_k (1-q_k+r_k)
    3. Add q_k + tau until this k is no longer the maximum
    4. Repeat steps 2,3 until sum(q) = 1
  
    
    Parameters:
    - r: 1D NumPy array of regularization terms

    Returns:
    - q: probability vector (same length as r) that solves the optimization

     

    """
    n = len(r)
    if np.sum(r) == 0:
        return np.ones(n) / n

    q = np.zeros(n)
    for _ in range(2 * n):
        rq = r - q
        rqmax = np.max(rq)
        S = rq == rqmax
        rq[S] = -np.inf
        k2 = np.argmax(rq)

        tau = rqmax - (r - q)[k2]
        tau = np.minimum(tau, (1 - np.sum(q)) / np.sum(S))
        q[S] += tau
        if np.sum(q) >= 1:
            break
        if np.sum(S) == n:
            tau = (1 - np.sum(q)) / np.sum(S)
            q[S] += tau
            break

    return q


def predict(q):

    """
    Samples a class label from a probability vector q using categorical sampling.

    If q is all zeros, the function samples uniformly.

    Parameters:
    - q: 1D array of class probabilities

    Returns:
    - y_pred: one-hot encoded vector representing the sampled class
    """
    if np.sum(q) == 0:
        sampled_index = np.random.choice(len(q))
    else:
        normalized_q = q / np.sum(q)
        sampled_index = np.random.choice(len(q), p=normalized_q)
    y_pred = np.zeros_like(q)
    y_pred[sampled_index] = 1

    return y_pred

 
     
