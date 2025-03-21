import numpy as np
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import laplacian
from numpy.linalg import pinv
import networkx as nx
import warnings
import random


def my_spmatrix_to_graph(adj_matrix):
    """
    Converts a SciPy sparse adjacency matrix to a NetworkX undirected graph.
    Node weights are preserved in the edge attributes.

    Parameters:
    - adj_matrix: SciPy CSR matrix (symmetric)

    Returns:
    - G: NetworkX undirected graph
    """
    G = nx.Graph()
    row, col = adj_matrix.nonzero()
    values = adj_matrix.data
    
    num_nodes = adj_matrix.shape[0]
    
    
    
    for i in range(num_nodes):
        G.add_node(i) 
     
    for ii in range(len(row)):
        i,j,v = row[ii],col[ii],values[ii]
        G.add_edge(i, j, weight=v)
     
    return G
def my_graph_to_spmatrix(G): 
    """
    Converts a NetworkX graph back to a SciPy sparse adjacency matrix in CSR format.

    Parameters:
    - G: NetworkX graph

    Returns:
    - csr_mat: SciPy sparse adjacency matrix
    """
    rows = []
    cols = []
    data = []
    
    for u, v, weight in G.edges(data='weight', default=None):
        rows.append(u)
        cols.append(v)
        data.append(weight)
        
        # Include the symmetric counterpart for undirected graphs
        if u != v:
            rows.append(v)
            cols.append(u)
            data.append(weight)
    
    # Create the CSR matrix
    csr_mat = csr_matrix((data, (rows, cols)))
    
    # Return the CSR matrix
    return csr_mat


def uniform_sparsify(adjacency, p, reweight = True, seed = None):
    """
    Sparsifies a graph by removing a random fraction of edges while preserving connectivity.

    Parameters:
    - adjacency: Input sparse adjacency matrix (CSR)
    - p: Fraction of edges to keep
    - reweight: Whether to rescale edge weights after sparsification
    - seed: Random seed for reproducibility

    Returns:
    - adjacency: Sparsified adjacency matrix
    - actual_p: Actual fraction of edges retained
    """
    
    if seed is not None:
        np.random.seed(seed)
    adjacency = triu(csr_matrix(adjacency))
    
    num_edges = adjacency.nnz
    orig_num_edges = num_edges + 0
    num_samples = int(num_edges * (1-p))
    
    actual_p = p + 0.
    num_edges_not_removed= None
    while num_samples > 0:
        row, col = adjacency.nonzero()
        values = adjacency.data

        # Calculate probabilities for each edge based on edge weights
        probabilities = values / np.sum(values)

        # Sample edges based on probabilities
        
        sampled_indices = np.random.choice(np.arange(num_edges), size=num_samples, replace=False, p=probabilities)

        # Extract sampled edges and their weights
        sampled_row = row[sampled_indices]
        sampled_col = col[sampled_indices]
        

        #graph = nx.from_scipy_sparse_array(adjacency + 0.)
        graph = my_spmatrix_to_graph(adjacency)
        num_edges_not_removed = 0

        for e in range(len(sampled_row)): 
            edge = (sampled_row[e], sampled_col[e])

            if graph.has_edge(*edge):
                w = graph[edge[0]][edge[1]]['weight']
                graph.remove_edge(*edge)

                if not nx.is_connected(graph):
                    graph.add_edge(edge[0],edge[1],weight=w)  # Add the edge back
                    num_edges_not_removed += 1

        adjacency = my_graph_to_spmatrix(graph)
        #adjacency = nx.to_scipy_sparse_array(graph)
        adjacency = triu(adjacency)
        if num_edges_not_removed == num_samples: 
            warnings.warn(f"Warning: max samples to be removed while keeping graph connected. {num_samples} not yet removed.")
            break
        num_samples = num_edges_not_removed
        num_edges = adjacency.nnz
        if num_samples > 0:
            warnings.warn(f"Warning: num_samples = {num_samples}, nnz = {adjacency.nnz}")

    adjacency = adjacency + adjacency.transpose()
    
    if reweight:
        if num_edges_not_removed is None:
            num_edges_removed = 0.
            actual_p = 1.
        else:
            num_edges_removed = orig_num_edges - num_edges_not_removed
            actual_p = num_edges_removed / orig_num_edges
            adjacency = adjacency / actual_p
    return adjacency, actual_p
    
     
def resistive_sparsify(adjacency, p, seed = None):
    """
    Sparsifies a graph based on resistive distances derived from the Laplacian pseudoinverse.

    Edges with lower resistive distances are more likely to be retained.

    Parameters:
    - adjacency: CSR-format adjacency matrix
    - p: fraction of edges to retain
    - seed: random seed for reproducibility

    Returns:
    - new_adjacency: sparsified adjacency matrix
    - actual_p: actual retained edge fraction
    """
    if seed is not None:
        np.random.seed(seed)
        
    adjacency.eliminate_zeros()
    if not (0 <= p <= 1):
        raise ValueError("The fraction p must be between 0 and 1.")
    
    laplacian = csgraph.laplacian(adjacency, normed=False).toarray()
    
    try:
        pseudoinverse = pinv(laplacian)
    except(np.linalg.LinAlgError):
        pseudoinverse = pinv(laplacian + .001*np.eye(laplacian.shape[0]))
        
    
    
    tadjacency = triu(adjacency)
    row, col = tadjacency.nonzero()
    values = tadjacency.data
    
    data = values * 0. 
    for e in range(len(row)):
        i = row[e]
        j = col[e]
        data[e] = pseudoinverse[i, i] + pseudoinverse[j, j] - 2 * pseudoinverse[i, j] 
    resistive_distances = sp.csr_matrix((data, (row, col)), shape=adjacency.shape)    
    resistive_distances.eliminate_zeros()
    
    new_adjacency, actual_p = uniform_sparsify(resistive_distances, p, reweight = False)
    
    
     
    for e in range(len(row)):
        i = row[e]
        j = col[e]
        if new_adjacency[i,j] == 0: continue
        new_adjacency[i,j] = adjacency[i,j] / actual_p
        new_adjacency[j,i] = adjacency[j,i] / actual_p
    return new_adjacency, actual_p




def influencer_sparsify(adjacency, qbar, seed = None):
    """
    Sparsifies a graph by limiting the degree of each node to a threshold qbar.

    Edges are randomly removed from high-degree nodes while preserving graph connectivity.

    Parameters:
    - adjacency: input adjacency matrix (CSR)
    - qbar: maximum degree allowed per node
    - seed: random seed

    Returns:
    - adjacency: sparsified adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
    #G = nx.from_scipy_sparse_array(adjacency)
    G = my_spmatrix_to_graph(adjacency)
    n = len(G.nodes)
    all_done = False
    while not all_done:
        total_removed = 0
        all_done = True
        rp = random.sample(range(n), n)
        nodes = list(G.nodes)
        nodes = [nodes[i] for i in rp]
        
        for ii, node in enumerate(nodes):
            remove = G.degree[node] - qbar
            du = G.degree[node] 
            if remove > 0:
                removed = 0
                neighbors = list(G.neighbors(node))
                rp = random.sample(range(len(neighbors)), len(neighbors))
                neighbors = [neighbors[i] for i in rp]
        
                for neighbor_node in neighbors:
                    w =  G.get_edge_data(node, neighbor_node)['weight']
                    G.remove_edge(node, neighbor_node)
                    if nx.is_connected(G):
                        removed += 1
                        total_removed += 1
                        if removed >= remove: break
                    else:
                        G.add_edge(node, neighbor_node, weight=w)
                if removed> 0:
                    for neighbor_node in G.neighbors(node):
                        w =  G.get_edge_data(node, neighbor_node)['weight']
                        G[node][neighbor_node]['weight'] =  (w / removed) * du
                
                if removed < remove:
                    all_done = False 
        if total_removed == 0:
            break
        else:
            print(total_removed)
    
    overflow = [G.degree[node] for node in G.nodes if G.degree[node] > qbar]
    
    if len(overflow) > 0:
        warnings.warn('was not able to reduce a row enough to achieve qbar')
        print(overflow)
            
    adjacency = my_graph_to_spmatrix(G)   
    return adjacency