import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type='ER',graph_params=None):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP
        graph_params (dict): params for graph_type

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_linear_sem(W, n,snr=None,noise_type='gauss'):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        noise_type (str): gauss, uniform
        SNR (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, snr):
      """X: [n, num of parents], w: [num of parents], x: [n]"""

      # Child node, use SNR, else scale 1
      if X.sum()!= 0:
        energy = np.mean(X**2)
        scale = 10**(-snr / 10)
        scale *= energy
        scale = np.sqrt(scale)
      else: scale = 1

      if noise_type == 'gauss':
        z = np.random.normal(scale=scale, size=n)
        x = X @ w + z
      elif noise_type == 'exp':
        z = np.random.exponential(scale=scale, size=n)
        x = X @ w + z
      elif noise_type == 'gumbel':
        z = np.random.gumbel(scale=scale, size=n)
        x = X @ w + z
      elif noise_type == 'uniform':
        z = np.random.uniform(low=-scale/2, high=scale/2, size=n)
        x = X @ w + z
      else:
          raise ValueError('unknown sem type')
      return x

    d = W.shape[0]
    if snr is None:
        snr_vec = np.ones(d) * float('inf')
    elif np.isscalar(snr):
        snr_vec = snr * np.ones(d)
    else:
        if len(snr) != d:
            raise ValueError('SNR must be a scalar or has length d')
        snr_vec = snr
    if not is_dag(W):
        raise ValueError('W must be a DAG')

    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], snr_vec[j])
    return X


def simulate_nonlinear_sem(B,n,sem_type,snr=None,noise_type='gauss'):
  """Simulate samples from nonlinear SEM.

  Args:
      B (np.ndarray): [d, d] binary adj matrix of DAG
      n (int): num of samples
      sem_type (str): mlp, mim, gp, gp-add

  Returns:
      X (np.ndarray): [n, d] sample matrix
  """
    
  def _simulate_single_equation(X, snr):
    """X: [n, num of parents], x: [n]"""

    # Child node, use SNR
    if X.sum()!= 0:
      energy = np.mean(X**2)
      scale = 10**(-snr / 10)
      scale *= energy
      scale = np.sqrt(scale)
    else: scale = 1

    if noise_type == 'gauss':
      z = np.random.normal(scale=scale, size=n)
    elif noise_type == 'exp':
      z = np.random.exponential(scale=scale, size=n)
    elif noise_type == 'gumbel':
      z = np.random.gumbel(scale=scale, size=n)
    elif noise_type == 'uniform':
      z = np.random.uniform(low=-scale/2, high=scale/2, size=n)
        
    # Non-linear model gen by type
    pa_size = X.shape[1]
    if pa_size == 0:
      return z
    if sem_type == 'mlp':
      hidden = 100
      W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
      W1[np.random.rand(*W1.shape) < 0.5] *= -1
      W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
      W2[np.random.rand(hidden) < 0.5] *= -1
      x = sigmoid(X @ W1) @ W2 + z
    elif sem_type == 'mim':
      w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w1[np.random.rand(pa_size) < 0.5] *= -1
      w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w2[np.random.rand(pa_size) < 0.5] *= -1
      w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w3[np.random.rand(pa_size) < 0.5] *= -1
      x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
    elif sem_type == 'gp':
        from sklearn.gaussian_process import GaussianProcessRegressor
        gp = GaussianProcessRegressor()
        x = gp.sample_y(X, random_state=None).flatten() + z
    elif sem_type == 'gp-add':
        from sklearn.gaussian_process import GaussianProcessRegressor
        gp = GaussianProcessRegressor()
        x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                  for i in range(X.shape[1])]) + z
    else:
        raise ValueError('unknown sem type')
    return x

  d = B.shape[0]
  if snr is None:
    snr_vec = np.ones(d) * float('inf')
  elif np.isscalar(snr):
    snr_vec = snr * np.ones(d)
  else:
    if len(snr) != d:
      raise ValueError('SNR must be a scalar or has length d')
    snr_vec = snr

  X = np.zeros([n, d])
  G = ig.Graph.Adjacency(B.tolist())
  ordered_vertices = G.topological_sorting()
  assert len(ordered_vertices) == d
  for j in ordered_vertices:
      parents = G.neighbors(j, mode=ig.IN)
      X[:, j] = _simulate_single_equation(X[:, parents], snr_vec[j])
  return X


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

def modify_DAG(true_DAG,num_changes):
    '''
    This function modifies the true DAG by adding or removing edges (flipping edged)
    
    Inputs:
        true_DAG: the true DAG
        num_changes: the total number of edges to add or remove
    '''
    if num_changes == 0: return true_DAG,0,0
    flip_mask = np.zeros_like(true_DAG)
    flip_mask[np.unravel_index(np.random.randint(0,true_DAG.shape[0] * true_DAG.shape[1],size=num_changes),true_DAG.shape)] = 1
    DAG_mod = np.abs(true_DAG - flip_mask)

    # Keep trying until mod is a DAG
    while (not ig.Graph.Adjacency(DAG_mod.tolist()).is_dag()  # DAG_mod is not a DAG
           or SHD(DAG_mod, true_DAG) != num_changes  # Hamming distance is correct. 
           or np.sum(DAG_mod.sum(axis=0) == 0) > np.sum(true_DAG.sum(axis=0) == 0)  # number of exos is not exceeded. 
          #  or np.sum((true_DAG - DAG_mod) > 0) == 0 # need atleast one loss
          ):
        # print(np.sum((true_DAG - DAG_mod) > 0) == 0)
        flip_mask = np.zeros_like(true_DAG)
        flip_mask[np.unravel_index(np.random.randint(0,true_DAG.shape[0] * true_DAG.shape[1],size=num_changes),true_DAG.shape)] = 1
        DAG_mod = np.abs(true_DAG - flip_mask)

    SHDN = np.sum((true_DAG - DAG_mod) > 0)
    SHDP = np.sum((true_DAG - DAG_mod) < 0)

    return DAG_mod,SHDN,SHDP


def modify_DAG_lossy(true_DAG,num_changes):
    '''
    This function modifies the true DAG by adding or removing edges (flipping edged)
    
    Inputs:
        true_DAG: the true DAG
        num_changes: the total number of edges to add or remove
    '''
    DAG_mod = true_DAG.copy()
    for i in range(num_changes):
        edges = np.where(DAG_mod)
        flip_i = np.random.randint(np.where(DAG_mod)[0].shape[0])
        DAG_mod[edges[0][flip_i],edges[1][flip_i]] = 0

    return DAG_mod

def SHD (B_true, B_est):
    '''
    Structural Hamming Distance
    
    Inputs:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_hypothesis (np.ndarray): 
    '''
    diff = np.abs(B_true - B_est)
    diff = diff + diff.transpose()
    diff[diff > 1] = 1  # Ignoring the double edges.
    return int(np.sum(diff)/2)