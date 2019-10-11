import numpy as np
import torch
from CoxphModel import TClassDataLoader


def unique_set(lifetime):
    a = lifetime.data.cpu().numpy()
    t, idx = np.unique(a, return_inverse=True)
    sort_idx = np.argsort(a)
    a_sorted = a[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return t, unq_idx

def acc_pairs(censor, lifetime):
    noncensor_index = np.nonzero(censor.data.cpu().numpy())[0]
    lifetime = lifetime.data.cpu().numpy()
    acc_pair = []
    for i in noncensor_index:
        all_j =  np.array(range(len(lifetime)))[lifetime > lifetime[i]]
        acc_pair.append([(i,j) for j in all_j])
    
    acc_pair = reduce(lambda x,y: x + y, acc_pair)
    return acc_pair

def log_parlik(lifetime, censor, score1):
    t, H = unique_set(lifetime)
    keep_index = np.nonzero(censor.data.cpu().numpy())[0]  #censor = 1
    H = [list(set(h)&set(keep_index)) for h in H]
    n = [len(h) for h in H]
    
    #score1 = score1.data.cpu().numpy()
    total = 0.0
    for j in range(len(t)):
        total_1 = torch.sum(torch.log(score1)[H[j]])
        m = n[j]
        total_2 = 0
        for i in range(m):
            subtotal = torch.sum(score1[sum(H[j:],[])]) - (i*1.0/m)*(torch.sum(score1[H[j]]))
            subtotal = torch.log(subtotal)
            total_2 = total_2 + subtotal
        total = total + total_1 - total_2
        #total = np.array([total])
    return torch.neg(total)

def c_index(censor, lifetime, score1):
    n_orderable = 0
    n_uncensor_pair = 0
    score_uncensor = 0
    score = 0
    score1 = score1.numpy().reshape(-1)
    for i in range(len(lifetime)):
        for j in range(i+1,len(lifetime)):
            if(censor[i] == 1 and censor[j] == 1):
                if(lifetime[i] > lifetime[j]):
                    n_orderable = n_orderable + 1
                    n_uncensor_pair += 1
                    if(score1[j] > score1[i]):
                        score = score + 1
                        score_uncensor += 1
                elif(lifetime[j] > lifetime[i]):
                    n_orderable = n_orderable + 1
                    n_uncensor_pair += 1
                    if(score1[i] > score1[j]):
                        score = score + 1
                        score_uncensor += 1
            elif(censor[i] == 1 and censor[j] == 0):
                if(lifetime[i] <= lifetime[j]):
                    n_orderable = n_orderable + 1
                    if(score1[j] < score1[i]):
                        score = score + 1
            elif(censor[j] == 1 and censor[i] == 0):
                if(lifetime[j] <= lifetime[i]):
                    n_orderable = n_orderable + 1
                    if(score1[i] < score1[j]):
                        score = score + 1
    #print(score_uncensor/n_uncensor_pair)
    return score / n_orderable

def evaluate_C_index(model, x, censor, lifetime):
    model.train(False)
    tloader = TClassDataLoader(x, censor, lifetime, batch_size = len(x))
    for step, (x, censor, lifetime) in enumerate(tloader):
        s = model(x).squeeze().data.detach()
        c = censor.squeeze().data.detach()
        l = lifetime.squeeze().data.detach()
        print("Score", s[0:10])
        print("Censor", c[0:10])
        print("Lifetime", l[0:10])
        c_i = c_index(c, l, s)
    
    model.train(True)
    return c_i

