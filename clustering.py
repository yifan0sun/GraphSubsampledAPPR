import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from appr import *
from tqdm import tqdm


def get_popular_seeds(cluster_map,num_clusters):
    """
    Selects the most 'popular' nodes as seeds for clustering.

    The cluster map is a matrix where each column corresponds to a 
    seed's influence vector (e.g., from APPR). This function sums 
    influence scores for each node across all seeds and picks the 
    top ones with highest aggregate influence.

    Parameters:
    - cluster_map: 2D matrix where column j contains influence scores from seed j
    - num_clusters: number of seeds to select

    Returns:
    - seeds: indices of the most popular nodes
    """
    n = cluster_map.shape[0]
    for i in range(n): cluster_map[i,i] = 0 
    popularity = np.array(cluster_map.sum(axis=1)).flatten()
    #pop_idx = np.argsort(popularity)[::-1]
    #seeds = pop_idx[:num_clusters]
    seeds = np.argpartition(popularity, -num_clusters)[-num_clusters:]

    return seeds

def get_cluster_map(adj_matrix, degree_vector, epsilon, alpha, sampling=None, maxneighbors=None,de_bias=False,de_bias_period=None,max_neighbors_debias=None ):

    """
    Constructs a matrix of approximate personalized PageRank (APPR) vectors for all nodes.

    Each column of the returned matrix corresponds to the APPR vector starting 
    from a seed node. These vectors will later be used to assign nodes to clusters.

    Parameters:
    - adj_matrix: sparse adjacency matrix of the graph
    - degree_vector: degree of each node
    - epsilon: APPR stopping threshold
    - alpha: teleport probability
    - sampling: optional sampling method (e.g., stoch_unif, det_topdegree)
    - maxneighbors: max number of neighbors to consider in APPR push steps
    - de_bias: whether to use dual correction to reduce subsampling error
    - de_bias_period: how frequently to apply dual correction
    - max_neighbors_debias: limit on neighbors used during correction

    Returns:
    - cluster_map: sparse matrix (n x n) where each column is an APPR vector
    """

    n = len(degree_vector)
    cluster_map = lil_matrix((n,n))
    for s in tqdm(range(n)):
        xs,r = appr(adj_matrix,degree_vector, sampling = sampling,  maxneighbors = maxneighbors,
                epsilon = epsilon, alpha= alpha, s = s,maxiter = 100,  seed = 42,
                de_bias = de_bias, de_bias_period = de_bias_period, max_neighbors_debias = max_neighbors_debias, 
                merge_weight=1 )
    
        idx = np.where(xs!=0)[0]
        if len(idx) > 0:
            cluster_map[idx,s] = xs[idx]
    return cluster_map



def cluster_ppr(cluster_map,seeds, labels,num_clusters, preprocessed = False):
    """
    Performs hard clustering of nodes using precomputed APPR vectors.

    Each node is assigned to the cluster corresponding to the seed 
    that gave it the highest influence score.

    Parameters:
    - cluster_map: matrix of APPR vectors (columns = seeds)
    - seeds: indices of seed nodes
    - labels: ground-truth labels for nodes (used to evaluate clustering accuracy)
    - num_clusters: number of clusters/seeds
    - preprocessed: if True, assumes cluster_map is already sliced and dense

    Returns:
    - score: fraction of nodes correctly clustered (based on dominant label match)
    - cluster_assign: vector of assigned cluster indices for each node
    """
    n = len(labels)
    
    if not preprocessed:
        for i in range(n): cluster_map[i,i] = 0 
        cluster_map = cluster_map[:,seeds]
        #cluster_assign = np.argmax(cluster_map, axis=1)
        try:
            cluster_map = cluster_map.toarray()
        except: pass
    cluster_assign = np.argmax(cluster_map, axis=1)


    #no_assign = np.sum(cluster_map,axis=1)
    no_assign = np.array(cluster_map.sum(axis=1)).flatten()
    cluster_assign[no_assign==0] = -1


    score = 0
    for i in range(num_clusters):
        gt_at_i = labels[cluster_assign==i]
        if len(gt_at_i) == 0:
            continue 
        most_common_label_count = np.max(np.bincount(gt_at_i))
        score += most_common_label_count
    return score/n,  cluster_assign
    