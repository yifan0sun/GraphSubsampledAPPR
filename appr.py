

import time
import numpy as np
from scipy.sparse import csr_matrix
from misc import get_top_value_random_lastelement


def find_neighbors(adj_matrix, node_index):
    """
    Returns the indices of nonzero neighbors of a node from the adjacency matrix.
    This is used to retrieve connected nodes efficiently.
    """
    node_row = adj_matrix[node_index]
    neighbors = node_row.nonzero()[1]
    return neighbors


def push(u,r,x,alpha,A,d, maxneighbors,  sampling, seed=None, update_x =True, de_bias = False, de_bias_period = 0, max_neighbors_debias = 0, merge_weight = 0.5  ):
    """
    Core operation used in APPR to update the residual (r) and solution (x) vectors.

    This function pushes probability mass from a single node u to its neighbors.
    If the node has too many neighbors, it samples a subset based on various strategies 
    (e.g., top degree, resistive score, edge weight).

    Parameters:
    - u: the node to push from
    - r: residual vector (gets updated)
    - x: solution vector (gets updated)
    - alpha: teleport probability in APPR
    - A: sparse adjacency matrix (CSR format)
    - d: degree vector
    - sampling: strategy for choosing neighbors if degree is too large
    - de_bias: whether to apply dual correction
    """
    seed = seed  % (2**32)
    np.random.seed(seed)

    def get_resistive(u, nu):

        z,_,_,_,_ = appr(A,d,.01, alpha, u,  maxneighbors = maxneighbors, maxiter = 5, sampling = 'stoch_unif', totrack = False,  seed = seed * 123456**2 + u*123456, de_bias = de_bias, de_bias_period = de_bias_period, max_neighbors_debias = max_neighbors_debias, merge_weight =merge_weight  )
        R = z[u] - 2*z
        for v in nu:
            z,_,_,_,_ = appr(A,d,.01, alpha, v,  maxneighbors = maxneighbors, maxiter = 5, sampling = 'stoch_unif', totrack = False ,  seed = seed * 123456**2 + u*123456 +v , de_bias = de_bias, de_bias_period =de_bias_period, max_neighbors_debias = max_neighbors_debias, merge_weight = merge_weight )
            R[v] += z[v]
        return R[nu]
        

 

    if d[u] == 0: return
    if update_x:
        x[u] = x[u] + alpha * r[u]
    nu = find_neighbors(A,u)
    
    
    if maxneighbors is not None and sampling is not None:
        if len(nu) > maxneighbors: 
            
            
            if sampling ==  'det_topweight':
                neighbors_vec = A[nu,u].toarray()[:,0]
                sidx = get_top_value_random_lastelement(neighbors_vec, maxneighbors)
                 
            elif sampling ==  'det_topresist':
                res = get_resistive(u,nu) 
                sidx = get_top_value_random_lastelement(res, maxneighbors)

            elif sampling ==  'det_topdegree': 
                neighbors_degree = np.array([np.sum(A[nuu,:].toarray()) for nuu in nu])
                sidx = get_top_value_random_lastelement(neighbors_degree, maxneighbors)
                
                
            elif sampling ==  'det_bottomdegree':
                neighbors_degree = np.array([np.sum(A[nuu,:].toarray()) for nuu in nu])
                sidx = get_top_value_random_lastelement(-neighbors_degree, maxneighbors)
                
            elif sampling ==  'stoch_topweight':
                neighbors_vec = A[nu,u].toarray()[:,0]
                sidx = np.random.choice(len(nu), maxneighbors, replace=False, p = neighbors_vec/np.sum(neighbors_vec))
                
            elif sampling ==  'stoch_topresist':
                res = get_resistive(u,nu)
                sidx = np.random.choice(len(nu), maxneighbors, replace=False, p = (res+.01)/np.sum(res+.01))
                 
                
            elif sampling ==  'stoch_topdegree':
                neighbors_degree = np.array([np.sum(A[nuu,:].toarray()) for nuu in nu])
                sidx = np.random.choice(len(nu), maxneighbors, replace=False, p = neighbors_degree/np.sum(neighbors_degree))
            elif sampling ==  'stoch_bottomdegree':
                neighbors_degree = np.array([np.sum(A[nuu,:].toarray()) for nuu in nu])
                
                neighbors_degree = np.max(neighbors_degree) - neighbors_degree+1
                sidx = np.random.choice(len(nu), maxneighbors, replace=False, p = neighbors_degree/np.sum(neighbors_degree))
            elif sampling ==  'stoch_unif':
                sidx = np.random.choice(len(nu), maxneighbors, replace=False)
                   
            nu = nu[sidx]
    
    #Theta = ((1+alpha)/2*np.eye(n) - (1-alpha)/2*A.toarray()@np.diag(sd) )/alpha
    
    neighbors_vec = A[nu,u].toarray()[:,0]
    if len(nu) > 0: 
        new_du = np.sum(neighbors_vec)
        if  update_x:  r[nu] = r[nu] + (1-alpha)*r[u]/(new_du*2)*neighbors_vec
        else: r[nu] = r[nu] + (1-alpha)*x[u]/(new_du*2)*neighbors_vec/ alpha
    
    if  update_x: r[u] = (1-alpha)*r[u]/2
    else: r[u] = r[u] -(1+alpha)*x[u] /(2* alpha)


             
def appr(A,d,epsilon, alpha, s, maxneighbors = None, maxiter = None, sampling = 'stoch_unif',  seed = None, de_bias = False, de_bias_period = 0, max_neighbors_debias = 0, merge_weight = 0.5 ):

    """
    Runs the Approximate Personalized PageRank (APPR) algorithm using push-based updates.

    Starts with a seed node s and computes an approximate solution vector x,
    along with a residual r that tracks approximation error.

    Includes optional dual-correction (de_bias) that improves accuracy
    even under heavy subsampling.

    Parameters:
    - A: sparse adjacency matrix
    - d: degree vector
    - epsilon: accuracy threshold
    - alpha: teleport probability
    - s: seed node
    - sampling: strategy to use for neighbor selection
    - maxiter: optional cap on iterations
    - de_bias: enables dual correction
    """
    
    n = A.shape[0]
    x = np.zeros(n)
    r = np.zeros(n)
    r[s] = 1.

    
    
    seed = seed  % (2**32)
    np.random.seed(seed)

 

    if de_bias:
        #weight = np.sqrt(d)
        weight = d / np.sum(d) 
        x_save = x+0.
        r_save = r+0.
        d_idx = np.argsort(d)[::-1]
    
    iter = 0
    while True:
        iter += 1
        
        S = np.where(np.abs(r) > epsilon*d)[0]
        
        if len(S) == 0:
            break
                
            
        for i in S:
            if r[i] > epsilon * d[i]:
                push(i,r,x,alpha,A,d,maxneighbors, sampling,  seed = seed * 123456 + iter)
        
        if de_bias:
            if iter % de_bias_period == 0:
                #de_bias_period = int(de_bias_period * 1.25)
                r_diff = r_save * 0.
         
                for u in np.where(x)[0]:
                     
                    push(u,r_diff,x,alpha,A,d,  max_neighbors_debias, sampling = sampling,   seed= seed * 1234561 + iter, update_x =False)
                 
                r_save  =  r_save + r_diff
                r = r_save + 0.
                
                x_save = x_save + x
                x = x * 0
                S = np.where(np.abs(r) > epsilon*d)[0]
               
        
        if maxiter is not None:
            if iter > maxiter:
                break
    
    if de_bias:
        x = x + x_save
    return x,r
 

def ppr_solve(A,d,alpha,y):
    """
    Solves the Personalized PageRank (PPR) linear system exactly via matrix inversion.

    This is used for comparison with the approximate methods.

    Parameters:
    - A: adjacency matrix
    - d: degree vector
    - alpha: teleport probability
    - y: source vector (usually one-hot at the seed node)
    """
    n = A.shape[0]
    sd = 1/d
    sd[np.isinf(sd)] = 0.
    Theta = ((1+alpha)/2*np.eye(n) - (1-alpha)/2*A.toarray()@np.diag(sd) )/alpha
    
    
    x = np.linalg.solve(Theta,y)
    
    r = Theta@x - y
    return x,r


def ppr_getres(A,d,alpha,x,y):
    """
    Computes the residual vector for a given solution x.

    Useful for evaluating the error of an approximate solution
    by comparing it to the expected result of the PPR linear system.

    Parameters:
    - x: approximate solution vector
    - y: source vector
    """
    n = A.shape[0]
    sd = 1/d
    sd[np.isinf(sd)] = 0.
    
    Thetax = x / alpha
    Thetax = (1+alpha)/2 * Thetax - (1-alpha)/2*(A@ (sd*Thetax))
     
    r = Thetax - y
    return r

 