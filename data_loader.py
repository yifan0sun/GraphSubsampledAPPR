import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
 

def load_graph_data(dataset):
    """
    Loads a graph's adjacency matrix and node labels from a preprocessed `.npz` file.

    Each file contains:
    - CSR-format sparse matrix data (data, indices, indptr, shape)
    - Labels array for nodes

    Parameters:
    - dataset: Name of the dataset (should match a file in graph_datasets/)

    Returns:
    - csr_mat: Sparse adjacency matrix in CSR format
    - labels: Array of node labels
    """
    
    f = np.load(f'graph_datasets/{dataset}.npz')
    csr_mat = csr_matrix(
        (f["data"], f["indices"], f["indptr"]), shape=f["shape"])
    csr_mat = csr_matrix(csr_mat)   
    csr_mat.eliminate_zeros() 

    labels = f['labels']
    return csr_mat, labels


 

def prepare_problems(csr_mat,labels):
    """
    Prepares matrices for downstream graph learning tasks.

    Computes:
    - Theta: the unnormalized graph Laplacian
    - normalized_Theta: symmetric normalized Laplacian
    - Y_onehot: one-hot encoding of node labels (sparse format)

    Parameters:
    - csr_mat: CSR-format sparse adjacency matrix
    - labels: Array of node labels

    Returns:
    - Theta: Unnormalized Laplacian (Deg - A)
    - normalized_Theta: Normalized Laplacian (D^{-1/2} (Deg - A) D^{-1/2})
    - Y_onehot: Sparse one-hot encoded label matrix (n_nodes × n_classes)
    """
    
    d = np.array(csr_mat.sum(axis=1)).flatten()
    Deg = csr_matrix((d, (range(len(d)), range(len(d)))), shape=(len(d), len(d)))
    Theta = Deg - csr_mat
    
    
    data = 1.0 / np.sqrt(d)
    indptr = np.arange(len(d) + 1)
    indices = np.arange(len(d))
    D_inv_sqrt = csr_matrix((data, indices, indptr), shape=(len(d), len(d)))
    
    normalized_Theta = D_inv_sqrt @ Theta @ D_inv_sqrt

    _, labels = np.unique(labels, return_inverse=True)
    num_classes = int(max(labels) + 1)
    row_indices = np.arange(len(labels))
    col_indices = np.array(labels)
    data = np.ones(len(labels))
    Y_onehot = csr_matrix(coo_matrix((data, (row_indices, col_indices)), shape=(len(labels), num_classes)))

    return Theta,normalized_Theta,  Y_onehot
