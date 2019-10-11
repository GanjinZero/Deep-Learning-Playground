import math
import numpy as np
import torch
import torch.nn as nn
from torchsummaryX import summary
from generate_fake_data import generate_01_data


class Coxph_model(nn.Module):
    def __init__(self, input_size_0, input_size_1, embedding_size=5, out_size=16):
        super(Coxph_model, self).__init__()
        self.embedding = nn.Embedding(input_size_0 + 1, embedding_size)
        self.activate = nn.ReLU()
        self.linear_1 = nn.Linear(input_size_1 * embedding_size, out_size)
        self.linear_2 = nn.Linear(out_size, 1)
        #self.beta = nn.Parameter(torch.Tensor(out_size + 1, 1))
        #self.beta.data.uniform_(-1/math.sqrt(out_size + 1), 1/math.sqrt(out_size + 1))
      
    
    """
    def score_2(self, score1):
        return self.sigmoid(self.fc_layer2(score1))
    """
    
    def forward(self, batch_1):
        s = self.embedding(batch_1)
        h = s.view(s.shape[0], -1)
        new_x = self.activate(self.linear_1(h))
        score1 = torch.exp(self.linear_2(new_x))
        return score1#, score2


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

        # NOTE: Use long() to make sure x input -> embedding
        seq_tensor_x = torch.zeros((len(batch_x), len(batch_x[0]))).long()
        seq_tensor_censored = torch.zeros(len(batch_censored),1).long()
        seq_tensor_y = torch.zeros(len(batch_y), 1)
        
        for idx, (seq_x, seq_censored, seq_y) in enumerate(zip(batch_x, batch_censored, batch_y)):
            seq_tensor_x[idx] = torch.tensor(seq_x)
            seq_tensor_censored[idx] = torch.tensor(seq_censored)
            seq_tensor_y[idx] = torch.tensor(seq_y)
        
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
            print(self.samples[i], self.censored_list[i], self.y_list[i])

    def report(self):
        print('# samples: {}'.format(len(self.samples)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))
        self.show_samples(n=5)

if __name__ == "__main__":
    # See Model Architeture
    summary(Coxph_model(input_size_0=100, input_size_1=20), torch.zeros(100, 20).long())

    # Test TClassDataLoader
    x, y = generate_01_data(500, 100, 10)
    censor = np.where(np.array(y)==1, 1, 0)
    tloader = TClassDataLoader(x, censor, y, batch_size=len(x))
    tloader.report()

