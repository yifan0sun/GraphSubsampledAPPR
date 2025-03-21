
import numpy as np
from misc import *
from appr import appr
from scipy.sparse import csr_matrix

from tqdm import tqdm

def PPR_for_ONL(adj_matrix,degree_vector,beta, sigma, t, epsilon, maxneighbors, maxiter, sampling, seed,
                de_bias = False, de_bias_period = 1, max_neighbors_debias = 1, merge_weight = 1):
    n = adj_matrix.shape[0]
    alpha = ((1-beta)*n+sigma)/((1+beta)*n+sigma)
    scale = ((1-beta)*n+sigma)/(2*n*sigma)

    pi,_ = appr(adj_matrix, degree_vector, epsilon, alpha, t,  maxneighbors , maxiter, sampling = sampling, seed = seed , de_bias = de_bias, de_bias_period = de_bias_period, max_neighbors_debias = max_neighbors_debias, merge_weight = merge_weight)
    sd = np.sqrt(degree_vector)
    x = pi / sd
    x = x * sd[t] / scale
    return x
        
def get_trace(adjmat,degrees,beta, sigma,  epsilon, maxneighbors, maxiter, sampling, seed,
              de_bias = False, de_bias_period = 1, max_neighbors_debias = 1, merge_weight = 1):
    tr = 0
    n = adjmat.shape[0] 
    for t in tqdm(range(n)):
        Mt =  PPR_for_ONL(adjmat,degrees,beta, sigma, t, epsilon, maxneighbors, maxiter, sampling = sampling, seed = seed*12345+t, de_bias = de_bias, de_bias_period = de_bias_period, max_neighbors_debias = max_neighbors_debias, merge_weight = merge_weight)
        tr += Mt[t]
    return tr
        
        

def Regularize_APPR(adjmat,degrees,epsilon, beta, sigma,  Y, seed = None, maxneighbors = None, maxiter = None, sampling = None,de_bias = False, de_bias_period = 10, max_neighbors_debias = 1, merge_weight = 1):
    n = adjmat.shape[0]
    n_classes = Y.shape[1] 
    Y = Y.toarray()
    
    track_y = []
    track_psi = []
    track_q = []
    
    G = np.zeros((n_classes, n))
    
    for t in tqdm(range(n)):
        
        #psi = -2 * (G @ M[:, t])
        Mt =  PPR_for_ONL(adjmat,degrees,beta, sigma, t, epsilon, maxneighbors, maxiter,sampling, seed = seed * 12345 + t,    de_bias = de_bias, de_bias_period = de_bias_period, max_neighbors_debias = max_neighbors_debias, merge_weight = merge_weight)
        psi = -2 * (G @ Mt)
        
             
        q = waterfill(psi)
        ypred = predict(q)
        # g = get_loss_grad(psi, q, Y[t, :])
        G[:, t] = -Y[t, :]

        track_y.append(ypred)
        track_psi.append(psi)
        track_q.append(q)

    track = {
            "y_pred": np.vstack(track_y),
            "psi": np.vstack(track_psi),
            "q": np.vstack(track_q)
        }    
    return track
 


def Relaxation_APPR(adjmat,degrees,epsilon, beta, sigma, D, Y,  maxneighbors = None, maxiter = None, sampling = None, seed = None,de_bias = False, de_bias_period = 10, max_neighbors_debias = 1, merge_weight = 1):
   

    def get_loss_grad(psi, q, y_true):
        if type(q) == csr_matrix: q = q.toarray()
        if type(y_true) == csr_matrix: y_true = y_true.toarray()
        if np.dot(y_true, q) == 0:

            i = np.argmax(psi)
            g = np.zeros_like(psi)
            g[i] = 1
            g = g - y_true
            g = g / (1 + 1 / sum(q > 0))

        else:
            qsupport = (q > 0) + 0
            g = -y_true + (qsupport - 1) / np.sum(qsupport)

        return g


    n = adjmat.shape[0]
    n_classes = Y.shape[1]
    

    track_y = []
    track_psi = []
    track_q = []

    #T = np.trace(M)
    T = get_trace(adjmat,degrees,beta, sigma,  epsilon, maxneighbors, maxiter, sampling, seed = seed ,  de_bias = de_bias, de_bias_period = de_bias_period, max_neighbors_debias = max_neighbors_debias, merge_weight = merge_weight)
    A = 0
    G = np.zeros((n_classes, n))
    for t in tqdm(range(n)):
        #if t % 100 == 0: print('\t rel', t,n)
        dem = np.sqrt(A + (D**2) * T)
        #psi = -2 * (G @ M[:, t])
        
        Mt =  PPR_for_ONL(adjmat,degrees,beta, sigma, t, epsilon, maxneighbors, maxiter, sampling, seed = seed * 12345 + t,  de_bias = de_bias, de_bias_period = de_bias_period, max_neighbors_debias = max_neighbors_debias, merge_weight = merge_weight) 
        psi = -2 * (G @ Mt)
         
                            
        if dem == 0:
            psi *= 0.0
        q = waterfill(psi)

        ypred = predict(q)
        g = get_loss_grad(psi, q, Y[t, :])
        G[:, t] = g
        A = A + 2 * np.dot(g, G @ Mt) + Mt[t] * np.linalg.norm(g) ** 2
        T = T - Mt[t]

        track_y.append(ypred)
        track_psi.append(psi)
        track_q.append(q)

    track = {
        "y_pred": np.vstack(track_y),
        "psi": np.vstack(track_psi),
        "q": np.vstack(track_q)
    }
    return track

 

 
 
