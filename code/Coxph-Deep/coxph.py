import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from generate_fake_data import generate_continue_data, generate_01_data
from CoxphModel import Coxph_model, TClassDataLoader 
from utils import *


def train(net, x, censor, lifetime, num_epochs, optimizer, batch_size):
    net.train(True)
    for epoch in range(num_epochs):
        total_loss = 0.0
        tloader = TClassDataLoader(x, censor, lifetime, batch_size = batch_size)
        for step, (x_batch, censor_batch, lifetime_batch) in enumerate(tloader):
            optimizer.zero_grad()
            score1 = net(x_batch)
            #print(score1)
            loss1 = log_parlik(lifetime_batch.squeeze(), censor_batch.squeeze(), score1)
            loss = loss1 #+ 0.5 * loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        mean_loss = total_loss / (int(len(x) / batch_size) + 1)
        if epoch==num_epochs or epoch % (num_epochs // 50) == 0:
            print(epoch, 'loss: ', mean_loss)
 

if __name__ == "__main__":
    # Parameters
    start_lr = 0.01
    l2_norm_loss_scale = 0.01
    epoch_nb = 100
    batch_size = 32

    # Test different data size, n=100, 500, 5000, 20000
    # p = 100, use_p = 10

    x, y = generate_01_data(500, 100, 10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    censor_train = np.where(np.array(y_train)==1, 0, 1) 
    censor_test = np.where(np.array(y_test)==1, 0, 1)

    model = Coxph_model(500, 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay = l2_norm_loss_scale)
   
    # Before train 
    train_C_index = evaluate_C_index(model, x_train, censor_train, y_train)
    test_C_index = evaluate_C_index(model, x_test, censor_test, y_test)
    print(f"Train C-index:{round(train_C_index, 2)}")
    print(f"Test C-index:{round(test_C_index, 2)}")

    # After train
    train(model, x_train, censor_train, y_train, epoch_nb, optimizer, batch_size)
    train_C_index = evaluate_C_index(model, x_train, censor_train, y_train)
    test_C_index = evaluate_C_index(model, x_test, censor_test, y_test)
    print(f"Train C-index:{round(train_C_index, 2)}")
    print(f"Test C-index:{round(test_C_index, 2)}")

