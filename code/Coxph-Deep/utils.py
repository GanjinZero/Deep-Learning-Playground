import numpy as np
import torch


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

def evaluate_C_index():
    print('Evaluate c index ...')
    model.train(False)
    tloader = TClassDataLoader(x_patient_train, x_treatment_train, censor_train, y_train, batch_size = len(x_patient_train))
    the_score1 = torch.zeros(len(x_patient_train), 1)
    the_censor = torch.zeros(len(x_patient_train))
    the_lifetime = torch.zeros(len(x_patient_train))
    for step,(x_patient, x_treatment, censor, lifetime) in enumerate(tloader):
        score1_train = model(x_patient, x_treatment)
        the_score1[step*len(x_patient_train):((step+1)*len(x_patient_train)),0] = score1_train.squeeze().data.detach()
        the_censor[step*len(x_patient_train):((step+1)*len(x_patient_train))] = censor.squeeze().data.detach()
        the_lifetime[step*len(x_patient_train):((step+1)*len(x_patient_train))] = lifetime.squeeze().data.detach()
            
        c_index_train = c_index(the_censor, the_lifetime, the_score1)
        print('Concordance index for training data: {:.4f}'.format(c_index_train))
    
    tloader = TClassDataLoader(x_patient_test, x_treatment_test, censor_test, y_test, batch_size = len(x_patient_test))
    for step,(x_patient, x_treatment, censor, lifetime) in enumerate(tloader):
        score1_test = model(x_patient, x_treatment)
        c_index_test = c_index(censor.squeeze(), lifetime.squeeze(), score1_test.squeeze())
        print('Concordance index for test data: {:.4f}'.format(c_index_test))
    model.train(True)
    print('Evaluate c index finish.')

