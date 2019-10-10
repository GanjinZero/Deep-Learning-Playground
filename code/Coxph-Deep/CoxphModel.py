import torch
import torch.nn as nn
from torchSummaryX import summary


class Coxph_model(nn.Module):
    def __init__(self, embedding_size = 1, out_size = 16):
        super(Coxph_model, self).__init__()
        self.embedding_1 = nn.Embedding(nb_patient + 1, embedding_size)
        self.embedding_2 = nn.Embedding(nb_treatment + 1, embedding_size)
        print('init done')
        # RELU
        #self.activate = nn.Sigmoid()
        #self.activate = nn.ReLU()
        #self.linear = nn.Linear(embedding_size, out_size)
        #self.fc_layer2 = nn.Linear(1, out_size)
        #self.beta = nn.Parameter(torch.Tensor(out_size, 1))
        #self.beta = nn.Parameter(torch.Tensor(out_size + 1, 1))
        #self.beta.data.uniform_(-1/math.sqrt(out_size + 1), 1/math.sqrt(out_size + 1))
        #print(self.linear.bias)
        #print(self.embedding_1.weight)
        
    def score_1(self, x):
        #score_1 = torch.exp(x.mm(self.beta))
        #print(score_1.shape)
        score_1 = torch.exp(x)
        return score_1
    
    """
    def score_2(self, score1):
        return self.sigmoid(self.fc_layer2(score1))
    """
    
    def forward(self, batch_1, batch_2):
        batch_1_h = self.embedding_1(batch_1)
        batch_2_h = self.embedding_2(batch_2)
        h = torch.cat((batch_1_h, batch_2_h), 1)
        s = torch.sum(h, 1)
        """
        new_x = self.activate(self.linear(s))
        new_x = torch.cat((new_x, torch.ones((new_x.shape[0], 1))), 1)
        score1 = self.score_1(new_x)
        """
        #score2 = self.score_2(score1)
        score1 = torch.exp(s)
        return score1#, score2

    
#test_model = Coxph_model()
#print(test_model(torch.zeros((3, 47), dtype=torch.long), torch.zeros((3, 25), dtype=torch.long)).shape)

"""
from torchsummaryX import summary
print(4)
summary(Coxph_model(),
        torch.zeros((3, 47), dtype=torch.long),
        torch.zeros((3, 25), dtype=torch.long))
print(5)
"""

class TClassDataLoader(object):
    def __init__(self, x, censored, time, batch_size=2):
        self.batch_size = batch_size
        self.samples = x
        self.censored_list = censored
        self.y_list = time

        self.shuffle_indices()
        self.n_batches = int(len(self.samples) / self.batch_size)

    def shuffle_indices(self):
        self.indices = np.random.permutation(len(self.samples))
        self.index = 0
        self.batch_index = 0

    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] =  batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch_x = []
        batch_censored = []
        batch_y = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch_x.append(self.samples[_index])
            batch_censored.append(self.censored_list[_index])
            batch_y.append(self.y_list[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor_x = torch.zeros((len(batch_x), len(batch_x[0]))).long()
        seq_tensor_censored = torch.zeros(len(batch_censored),1).long()
        seq_tensor_y = torch.zeros(len(batch_y), 1).long()
        
        for idx, (seq_x, seq_censored, seq_y) in enumerate(zip(batch_x, batch_censored, batch_y)):
            seq_tensor_x[idx] = torch.tensor(seq_x).long()
            seq_tensor_censored[idx] = torch.tensor(seq_censored).long()
            seq_tensor_y[idx] = torch.tensor(seq_y).long()
        
        return seq_tensor_x, seq_tensor_censored, seq_tensor_y

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:#stop iteration
                raise StopIteration()
            yield self._create_batch() # return tensor[] and corresponding length

    def show_samples(self, n=10):
        for i in range(n):
            print(self.samples_x[i], self.censored_list[i], self.y_list[i])

    def report(self):
        print('# samples: {}'.format(len(self.samples_x)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))
        self.show_samples(n=3)

