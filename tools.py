import numpy as np
# import pandas as pd
# import anndata as ad
# import scanpy as sc
# from sklearn.preprocessing import normalize
import torch
from torch.utils.data import Dataset
import scipy.io
import scipy.sparse as sp
import random as random
from torch.nn.functional import normalize


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed:',seed)

def normalize_rows_zscore(x):
    x = torch.tensor(x)
    mean = torch.mean(x, dim=0, keepdim=True)
    std = torch.std(x, dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-12)
    xnorm = (x - mean) / std
    xnorm = np.array(xnorm, dtype=np.float32)
    return xnorm     

def load_data(path, labelpath):
    dataset = Diabets(path, labelpath)
    f1, f2 = dataset.x1.shape[1], dataset.x2.shape[1]
    dims = [f1, f2]
    view = len(dims)
    data_size = dataset.__len__()
    class_num = len(np.unique(dataset.y)) 
    
    return dataset, dims, view, data_size, class_num


from sklearn.decomposition import PCA
class Diabets(Dataset):
    def __init__(self, path, labelpath):
        data1 = scipy.io.loadmat(path)['X1'].astype(np.float32)  # gut_16s  moid: genera, met;   
        data2 = scipy.io.loadmat(path)['X2'].astype(np.float32)  # RNAseq   GC/ESRD: X1, X2

        labels = scipy.io.loadmat(labelpath)['label']

        # data2 = np.log(data2+1)   # for MOID
        # data2 = data2 / data2.sum(axis=1).reshape(-1,1)  # for MOID
               
        # labels = labels.astype(np.float32)
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        dataview = [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])]
       
        return dataview, torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_view(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)

"""
KNN adjacency matrix calculation
"""
def create_sparse(I):

    similar = I.reshape(-1).tolist()
    index = np.repeat(range(I.shape[0]), I.shape[1])

    assert len(similar) == len(index)
    indices = torch.tensor([index, similar])
    result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)),
                                     [I.shape[0], I.shape[0]])

    return result
    

"""
Calculation of distance matrix in the formula
"""
def pairwise_distance(x, y=None):
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1) # [B, N, f]
    A = -2 * torch.bmm(y, x) # [B, N, N]
    A += torch.sum(y**2, dim=2, keepdim=True) # [B, N, 1]
    A += torch.sum(x**2, dim=1, keepdim=True) # [B, 1, N]
    return A.squeeze()

    
def get_Gauss_Similarity(interaction_matrix):
    # row: instance  col: features
    interaction_matrix = np.array(interaction_matrix)
    X = np.mat(interaction_matrix)
    delta = 1 / np.mean(np.power(X,2), 0).sum()
    alpha = np.power(X, 2).sum(axis=1)
    result = np.exp(np.multiply(-delta, alpha + alpha.T - 2 * X * X.T))
    # similarity_matrix[np.isnan(similarity_matrix)] = 0
    # result = result - np.diag(np.diag(result))
    result = torch.tensor(result)
    return result
    

def nndsvd(A, k):
	'''
	This function implements the NNDSVD algorithm described in [1] for
	sinitializattion of Nonnegative Matrix Factorization Algorithms.

	Parameters
	------------
	 A    : the input nonnegative m x n matrix A
	 k    : the rank of the computed factors W,H
	 flag : indicates the variant of the NNDSVD Algorithm
			flag=0 --> NNDSVD
			flag=1 --> NNDSVDa
			flag=2 --> NNDSVDar
	Returns
	 -------------
	 W   : nonnegative m x k matrix
	 H   : nonnegative k x n matrix

	 References:
	 [1] C. Boutsidis and E. Gallopoulos, SVD-based initialization: A head
		 start for nonnegative matrix factorization, Pattern Recognition
    '''

	#check the input matrix
	if A.min() < 0:
		raise ValueError('The input matrix contains negative elements !')

	#size of the input matirx
	m = A.size(0)
	n = A.size(1)

	#the matrices of the factorization
	try:
		W = torch.zeros((m,k)) # .cuda()
		H = torch.zeros((k,n)) # .cuda()
	except:
		W = torch.zeros((m,k))
		H = torch.zeros((k,n))

	#1st SVD --> partial SVD rank-k to the input matrix A.
	U, S, V = torch.svd(A)
	#print('U dtype',U.dtype)
	
	W[:,0] = torch.sqrt(S[0]) * torch.abs(U[:,0])
	H[0,:] = torch.sqrt(S[0]) * torch.abs(V[:,0])

	U = U[:,1:k]
	V = V[:,1:k]
	S = S[1:k]

	U_p = U.clone()
	U_p[U_p < 0] = 0

	U_n = U
	U_n[U_n > 0] = 0
	U_n = -U_n

	V_p = V.clone()
	V_p[V_p < 0] =0

	V_n = V
	V_n[V_n > 0] = 0
	V_n = -V_n

	norm_U_p = torch.norm(U_p,dim = 0)
	norm_U_n = torch.norm(U_n,dim = 0)
#     print('norm_U_p', norm_U_p)
#     print('norm_U_n',norm_U_n)

	norm_V_p = torch.norm(V_p,dim = 0)
	norm_V_n = torch.norm(V_n,dim = 0 )
#     print('norm_V_p',norm_V_p)
#     print('norm_V_n',norm_V_n)

	termp = norm_U_p * norm_V_p
	termn = norm_U_n * norm_V_n

	tmp_mul = torch.sqrt(S*termp)[None,:]
	W_p = tmp_mul * U_p / norm_U_p[None,:]
	H_p = tmp_mul * V_p / norm_V_p[None,:]
	H_p = H_p.T

	tmp_mul = torch.sqrt(S*termn)[None,:]
	W_n = tmp_mul * U_n / norm_U_n[None,:]
	H_n = tmp_mul * V_n / norm_V_n[None,:]
	H_n = H_n.T

	W[:,1:] = W_p
	H[1:,:] = H_p
	# print('W dtype',W.dtype)
	# print('W_n.dtype',W_n.dtype)
	ind_n = termp < termn
	W[:,1:][:,ind_n] = W_n[:,ind_n]
	H[1:,:][ind_n,:] = H_n[ind_n,:]
#     print(H[1:,:][ind_n,:] )
#     print(H_n[ind_n,:])

	eps = 0.0000000001

	W[W<eps] = 0
	H[H<eps] = 0

	return W, H


def dist2(x,c):
	'''
	Calculates the squared Euclidean distance between two matrices
	Parameters
	------------
	 x    : (M,N) matrix torch.tensor
	 c    : (L,N) matrix torch.tensor
	Returns
	 -------------
	res   : (M,L) matrix torch.tensor
	'''
	ndata,dimx = x.size()
	ncentres, dimc = c.size()

	if dimx != dimc:
		raise ValueError('Data dimension does not match dimension of centres')
	try:
		tmp_1 = torch.ones(ncentres,1)            # .cuda()
		tmp_2 = torch.ones(ndata,1)               # .cuda()
	except:
		tmp_1 = torch.ones(ncentres,1)
		tmp_2 = torch.ones(ndata,1)

	part1 = tmp_1 @ torch.sum(torch.square(x).T,0)[None,:]
	part2 = tmp_2 @  torch.sum(torch.square(c).T,0)[None,:]
	part3 = 2 * x@c.T

	del tmp_1, tmp_2
	torch.cuda.empty_cache()
   
	res = part1.T + part2 - part3
	return res 

from torch.distributions import Normal
def affinityMatrix(Diff,K = 20,sigma = 0.5):
	r"""
	Calculates affinity matrix given distance matrix
	Uses a scaled exponential similarity kernel to determine the weight of each
    edge based on `dist`. Optional hyperparameters `K` and `mu` determine the
    extent of the scaling (see `Notes`).

	Parameters
    ----------
    Diff : (N, N) array_like
        Distance matrix
    K : (0, N) int, optional
        Number of neighbors to consider. Default: 20
    mu : (0, 1) float, optional
        Normalization factor to scale similarity kernel. Default: 0.5

    Returns
    -------
    W : (N, N) torch.tenosr
        Affinity matrix

	Notes
    -----
    The scaled exponential similarity kernel, based on the probability density
    function of the normal distribution, takes the form:

    .. math::

       \mathbf{W}(i, j) = \frac{1}{\sqrt{2\pi\sigma^2}}
                          \ exp^{-\frac{\rho^2(x_{i},x_{j})}{2\sigma^2}}

    where :math:`\rho(x_{i},x_{j})` is the Euclidean distance (or other
    distance metric, as appropriate) between patients :math:`x_{i}` and
    :math:`x_{j}`. The value for :math:`\\sigma` is calculated as:

    .. math::

       \sigma = \mu\ \frac{\overline{\rho}(x_{i},N_{i}) +
                           \overline{\rho}(x_{j},N_{j}) +
                           \rho(x_{i},x_{j})}
                          {3}

    where :math:`\overline{\rho}(x_{i},N_{i})` represents the average value
    of distances between :math:`x_{i}` and its neighbors :math:`N_{1..K}`,
    and :math:`\mu\in(0, 1)\subset\mathbb{R}`.

	"""

	eps = 2.2204e-16
	Diff = (Diff + Diff.T)/2
	Diff = Diff - torch.diag(torch.diag(Diff))

	T = Diff.sort(dim = 1)[0]
	m, n = Diff.size()
	try: 
		W = torch.zeros((m,n)).cuda()
	except:
		W = torch.zeros((m,n))
	TT = T[:,1: K+1].mean(dim = 1) + eps
	Sig =  (TT[:,None] + TT[None,:] + Diff ) / 3

	Sig[Sig<=eps] = eps
	
	W = Normal(0 , sigma * Sig).log_prob(Diff).exp()
	del Sig, Diff
	torch.cuda.empty_cache()

	W = (W + W.T)/2
	return W


def BOnormalized(W, alpha = torch.tensor(1)):
	"""
    Adds `alpha` to the diagonal of `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF
    alpha : (0, 1) torch.tensor, optional
        Factor to add to diagonal of `W` to increase subject self-affinity.
        Default: 1.0

    Returns
    -------
    W : (N, N) torch.tensor
        Normalized similiarity array
    """

	try:
		tmp = torch.eye(W[0].size(0))   # .cuda()
		alpha = alpha                   # .cuda()
	except:
		tmp = torch.eye(W[0].size(0))

	W = W + alpha * tmp
	del tmp
	torch.cuda.empty_cache()

	W = (W +W.transpose(1,2)) / 2

	return W

def FindDominateSet(W,K = 20):
	"""
    Retains `K` strongest edges for each sample in `W`

    Parameters
    ----------
    W : (N, N) array_like
        Input data
    K : (0, N) int, optional
        Number of neighbors to retain. Default: 20

    Returns
    -------
    Wk : (N, N) torch.tensor
        Thresholded version of `W`
    """

	m,n = W.size()
	_,indices = torch.sort(W,1,descending = True)
	try:
		tmp = torch.arange(n)[:,None]          # .cuda()
		newW = torch.zeros((m,n))              # .cuda()
	except:
		tmp = torch.arange(n)[:,None]
		newW = torch.zeros((m,n))
	#newW = torch.zeros((m,n))
	keeped_col = indices[:,:K]
	newW[tmp,keeped_col] = W[tmp,keeped_col]
	newW = newW / newW.sum(1)[:,None]

	return newW


def SNF(*Wall, K = 20, t = 20, alpha = torch.tensor(1)):
	'''
	This function implements the SNF algorithm described in [1] for data integration.

	Parameters
	----------
	*aff : (N, N) array_like, torch.tensor
		Input similarity arrays; all arrays should be square and of equal size.
	K : (0, N) int, optional
		Hyperparameter normalization factor for scaling. Default: 20
	t : int, optional
		Number of iterations to perform information swapping. Default: 20
	alpha : (0, 1) torch.tensor, optional
		Hyperparameter normalization factor for scaling. Default: 1.0

	Returns
	-------
	W: (N, N) torch.tensor
		Fused similarity network of input arrays
	'''
	Wall = torch.stack(list(Wall), dim=0)
	C = Wall.size(0)
	m,n = Wall[0].size()

	try:
		newW = torch.empty((C,m,n))    #.cuda()
	except:
		newW = torch.empty((C,m,n))


	for i in range(C):
		Wall[i] = Wall[i] / Wall[i].sum(1)[:,None]
		Wall[i] = (Wall[i] + Wall[i].T)/2
		newW[i] = FindDominateSet(Wall[i],K)


	Wsum = Wall.sum(dim = 0)
	# del newW
	torch.cuda.empty_cache()

	for iter in range(t):
		#for i in range(C):
		Wall0 = newW @ (Wsum - Wall) @ newW.transpose(1,2) / (C - 1)
		Wall = BOnormalized(Wall0,alpha)
		Wsum = Wall.sum(0)
	

	del Wall0, Wall, newW
	torch.cuda.empty_cache()

	W = Wsum / C
	W = W / W.sum(1)[:,None]
	try:
		tmp = torch.eye(n)   # .cuda()
	except:
		tmp = torch.eye(n)
	W = (W + W.T + tmp) / 2

	return W


def top_sim(A, k=10):
    """
    A : torch tensor
    k : neghbor number, The default is 10.

    Returns
    A_new : fetch topk largest values in each row of A
    """
    
    result = torch.zeros_like(A)
    values, indices = torch.topk(A, k, dim=1)

    rows = torch.arange(A.size(0)).unsqueeze(1).expand(-1, k)
    result[rows, indices] = values

    A_new = (result+result.T)/2
    # A_new = result

    return A_new


def param_init(dataset, view, k):     
    n = len(dataset)
    index = torch.arange(n)
    xs, _, _ = dataset.__getitem__(index)
   
    D = []
    W = []
    H = []
    S = []
    DD = []
    Diff = []
    
    for v in range(view):     
        D.append(pairwise_distance(xs[v]))  # Calculate distance matrix D
        D[v] = normalize(D[v])     
                
        dif = dist2(xs[v], xs[v])
        Diff.append(dif)
        S.append(affinityMatrix(Diff[v],K = 20))
        DD.append(torch.diag(S[v].sum(0)))
        
        xs[v] = xs[v].T       
        xs[v][xs[v] < 0] = 0        
        factor_W, factor_H = nndsvd(xs[v], k)
        W.append(factor_W)
        H.append(factor_H)    
    
    # A = 1-(D[0] + D[1])/2
    A = SNF(S[0], S[1], K = 20)
    
    G = []
    t0 = cos_sim(H[0].T, H[0].T)
    G.append(t0)
    
    t1 = cos_sim(H[1].T, H[1].T)
    G.append(t1)   
    G_ = A
    
    allones = torch.ones(n,1)
    X1, X2 = torch.tensor(xs[0]), torch.tensor(xs[1])
    W1, W2 = torch.tensor(W[0]), torch.tensor(W[1])
    H1, H2 = torch.tensor(H[0]), torch.tensor(H[1])
    A1, A2, A_ = torch.tensor(G[0]), torch.tensor(G[1]), torch.tensor(G_)
    L1, L2 = DD[0]-S[0], DD[1]-S[1]
   
    err1 = torch.square(torch.norm(X1@(A_+A1) - W1@H1)) + torch.square(torch.norm(X2@(A_+A2) - W2@H2))
    err2 = (torch.square(torch.norm((A_+A1) - H1.T@H1)) +  torch.square(torch.norm((A_+A2) - H2.T@H2)))
    err3 = torch.sum(torch.log(H1+1)) + torch.sum(torch.log(H2+1))   
    err4 = (H1@L1@H1.T + H2@L2@H2.T).trace()
    err5 = torch.square(torch.norm(A1@allones - allones)+torch.norm(A2@allones - allones)+torch.norm(A_@allones - allones))
        
    alpha = err1 / err2
    gamma = torch.abs(err1 / err4)
    phi = err1 / err3
    yita = err1 / err5    
    
    alpha = alpha/5
    gamma = gamma / 1e+2
    phi = 0.5*phi/1e+4   
    yita= yita
    return alpha, gamma, yita, phi


def cos_sim(mat1: torch.Tensor, mat2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 归一化
    mat1_norm = mat1 / (mat1.norm(dim=1, keepdim=True) + eps)
    mat2_norm = mat2 / (mat2.norm(dim=1, keepdim=True) + eps)

    similarity_matrix = torch.mm(mat1_norm, mat2_norm.t())   
    return similarity_matrix
