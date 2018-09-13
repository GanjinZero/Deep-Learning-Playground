# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:10:35 2018

@author: GanJinZERO
"""


import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

def check_valid_move(state):
    if state[0] > 4:
        return False
    if state[0] < 0:
        return False
    if state[1] > 4:
        return False
    if state[1] < 0:
        return False
    return True

R = [[-1,-1,-1,-1,-1],
     [-1,-100,-1,-100,-1],
     [-1,-1,-1,-1,-1],
     [-1,-100,-1,-100,-1],
     [-1,-1,100,-1,-1]]

epoch_count = 0
move_mode = [[1, 0], [0, 1], [-1, 0], [0, -1]]

class DQNagent:
    def __init__(self):
        self.big_integer = 10000000
        self.state_size = 2
        self.epsilon = 0.2
        self.lbd = 0.8
        self.learning_rate = 0.01
        self.memory = deque(maxlen = 100)
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_shape = (self.state_size, ), activation='relu'))
        model.add(Dense(16, activation = 'relu'))
        model.add(Dense(4, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        # This is a Q_a(s) model, we can also build a Q(s,a) model.
        return model
    
    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        
    def act(self, state, tp=0):
        use_number = []
        not_use_number = []
        for i in range(4):
            new_state_x = state[0] + move_mode[i][0]
            new_state_y = state[1] + move_mode[i][1]
            if check_valid_move([new_state_x, new_state_y]):
                use_number.append(i)
            else:
                not_use_number.append(i)
        u = np.random.rand()
        if tp == 0 and u <= self.epsilon:
            return (use_number[np.random.randint(0, len(use_number))])
        else:
            Q_use = self.model.predict(np.array([state]))[0]
            for nb in not_use_number:
                Q_use[nb] = -self.big_integer
            return np.argmax(Q_use)
        
    def update(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        counter = 0
        state_train = np.array([])
        target_train = np.array([])
        for state, action, reward, next_state, terminal in minibatch:
            counter += 1
            y = reward
            if not terminal:
                Q_next = self.model.predict(np.array([next_state]))[0]
                for i in range(4):
                    new_state_x = state[0] + move_mode[i][0]
                    new_state_y = state[1] + move_mode[i][1]
                    if not check_valid_move([new_state_x, new_state_y]):
                        Q_next[i] = -self.big_integer
                y += self.lbd * max(Q_next)
            target_use = self.model.predict(np.array([state]))
            target_use[0][action] = y
            if counter == 1:
                state_train = np.array([state])
                target_train = np.array(target_use)
            else:
                state_train = np.concatenate((state_train, np.array([state])), axis = 0)
                target_train = np.concatenate((target_train, np.array(target_use)), axis = 0)
        self.model.fit(state_train, target_train, epochs=1, verbose=0)
        # That's not what minibatch do! This is min batch.

    def draw_answer(self, state):
        dir_mode = ["↓", "→", "↑", "←"]
        picture = [["R","R","R","R","R"],
                   ["R","W","R","W","R"],
                   ["R","R","R","R","R"],
                   ["R","W","R","W","R"],
                   ["R","R","O","R","R"]]
        while R[state[0]][state[1]] != 100:
            action = self.act(state, tp = 1)
            picture[state[0]][state[1]] = dir_mode[action]
            state = [state[0] + move_mode[action][0],
                     state[1] + move_mode[action][1]]
        for line in picture:
            print("".join(line))

if __name__ == "__main__":
    
    agent = DQNagent()
    batch_size = 1
    # If batch_size is big, there seems some speed performance issues.
    step_count_sum = 0
    step_count = 0
    print("Agent init done.")
    
    while epoch_count < 1000:
        epoch_count += 1
        step_count_sum += step_count
        step_count = 0
        print(epoch_count)
        print(step_count_sum)
        state = np.array([random.randint(0, 4), random.randint(0, 4)])
        while R[state[0]][state[1]] != 100 and step_count < 200:
            step_count += 1
            action = agent.act(state)
            next_state = np.array([state[0] + move_mode[action][0],
                          state[1] + move_mode[action][1]])
            reward = R[next_state[0]][next_state[1]]
            terminal = False
            if R[state[0]][state[1]] == 100:
                terminal = True
            agent.remember(state, action, reward, next_state, terminal)
            if len(agent.memory) > batch_size:
                agent.update(batch_size)
            state = next_state
                
    print(epoch_count)
    agent.draw_answer([0, 1])            