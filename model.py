from tqdm import tqdm
from sklearn.cluster import KMeans
from tools import *


def cos_sim(mat1: torch.Tensor, mat2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mat1_norm = mat1 / (mat1.norm(dim=1, keepdim=True) + eps)
    mat2_norm = mat2 / (mat2.norm(dim=1, keepdim=True) + eps)

    similarity_matrix = torch.mm(mat1_norm, mat2_norm.t())   
    return similarity_matrix

def JointNMF(dataset, sim_LLM, view, k, alpha, gamma, yita, phi):     
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
    
    A = SNF(S[0], S[1], K = 20)
    
    G = []
    t0 = cos_sim(H[0].T, H[0].T)
    G.append(t0)
    
    t1 = cos_sim(H[1].T, H[1].T)
    G.append(t1)
     
    G_ = A
    
    # introduce LLM embeds
    sim_LLM = torch.tensor(sim_LLM)
    S[0] = sim_LLM
    DD[0] = torch.diag(S[0].sum(0))
       
    obj_old = torch.tensor(1.)
    Maxiter = 300
    tot_loss = 0.
    for epoch in range(Maxiter):       
        # update W
        XHt0 = torch.mm(xs[0], (G_ + G[0]))
        XHt0 = torch.mm(XHt0, H[0].T)
        HHt0 = torch.mm(H[0], H[0].T)
        WHHt0 = torch.mm(W[0], HHt0)
        WHHt0[WHHt0 < 1e-10] = 1e-10
        W[0] = W[0] * (XHt0 / WHHt0)
    
        XHt1 = torch.mm(xs[1], (G_ + G[1]))
        XHt1 = torch.mm(XHt1, H[1].T)
        HHt1 = torch.mm(H[1], H[1].T)
        WHHt1 = torch.mm(W[1], HHt1)
        WHHt1[WHHt1 < 1e-10] = 1e-10
        W[1] = W[1] * (XHt1 / WHHt1) 
        
        # update H
        Numerator1 = W[0].T @ xs[0] @ (G_+G[0])  + 2*alpha*H[0]@(G_.T+G[0].T) + gamma*(H[0]@S[0]) + phi*(torch.sum(torch.log(H[0]+1))+n*k) #+n*k
        denominator1 = (W[0].T @ W[0])@H[0] + 2*alpha*(H[0]@H[0].T)@H[0] + gamma*(H[0]@DD[0])
        denominator1[denominator1 < 1e-10] = 1e-6  # avoiding Nan in iteration
        H[0] = H[0] * (Numerator1 / denominator1)
                
        Numerator2 = W[1].T @ xs[1] @ (G_+G[0]) + 2*alpha*H[1]@(G_.T+G[0].T) + gamma*(H[1]@S[1]) + phi*(torch.sum(torch.log(H[1]+1))+n*k) #+n*k)
        denominator2 = (W[1].T @ W[1])@H[1] + 2*alpha*(H[1]@H[1].T)@H[1] + gamma*(H[1]@DD[1])
        denominator2[denominator2 < 1e-10] = 1e-6
        H[1] = H[1] * (Numerator2 / denominator2)
        
        # update A        
        nume_A1 = xs[0].T@(W[0]@H[0]) + alpha*(H[0].T@H[0]) + yita*torch.ones(n,n)
        denume_A1 = (xs[0].T@xs[0]) @ (G_+G[0]) + alpha*(G_+G[0]) + yita*torch.ones(n,n)*G[0]
        denume_A1[denume_A1 < 1e-10] = 1e-6
        G[0] = G[0] * (nume_A1 / denume_A1)
        
        nume_A2 = xs[1].T@(W[1]@H[1]) + alpha*(H[1].T@H[1]) + yita*torch.ones(n,n)
        denume_A2 = (xs[1].T@xs[1]) @ (G_+G[1]) + alpha*(G_+G[1])+ yita*torch.ones(n,n)*G[1]
        denume_A2[denume_A2 < 1e-10] = 1e-6
        G[1] = G[1] * (nume_A2 / denume_A2)
        
        Numerator = xs[0].T@(W[0]@H[0]) + xs[1].T@(W[1]@H[1]) + alpha*(H[0].T@H[0] + H[1].T@H[1]) + yita*torch.ones(n,n)
        denominator = (xs[0].T@xs[0]) @ (G_+G[0]) + (xs[1].T@xs[1]) @ (G_+G[1]) + alpha*(2*G_+G[0]+G[1]) + yita*torch.ones(n,n)*G_
        denominator[denominator < 1e-10] = 1e-6
        G_ = G_ * (Numerator / denominator)
        
        obj = compute_obj(xs, W, H, G, G_, S, DD, alpha, gamma, yita, phi)
        error = torch.abs(obj_old - obj) / obj_old
        if  (error < 1e-6 and epoch > 0) or epoch == Maxiter - 1:
            print('number of epoch:', epoch+1)
            print('obj:',obj)
            break
        print(f'Epoch {epoch}')
        
    H1, H2 = H[0].T, H[1].T
    W1, W2 = W[0], W[1]

    return W1, W2, H1, H2, G_, G


def compute_obj(xs, W, H, G, G_, S, DD, alpha, gamma, yita, phi):
    ''' function to comupte the objective to be optimized '''
    L1, L2 = DD[0]-S[0], DD[1]-S[1]
    X1, X2 = xs[0], xs[1]
    W1, W2 = W[0], W[1]
    H1, H2 = H[0], H[1]
    A1, A2, A_ = G[0], G[1], G_
    
    n = G_.shape[0]
    
    obj = torch.square(torch.norm(X1@(A_+A1) - W1@H1)) + torch.square(torch.norm(X2@(A_+A2) - W2@H2))
    obj = obj + alpha *(torch.square(torch.norm((A_+A1) - H1.T@H1)) +  torch.square(torch.norm((A_+A2) - H2.T@H2)))
    obj += gamma * (torch.trace(H1 @ L1 @ H1.T + H2 @ L2 @ H2.T))
    allones = torch.ones(n,1)
    obj += yita * torch.square(torch.norm(A1@allones - allones)+torch.norm(A2@allones - allones)+torch.norm(A_@allones - allones))
    obj += phi * torch.sum(torch.log(H1+1)) + torch.sum(torch.log(H2+1))

    return obj
    
    