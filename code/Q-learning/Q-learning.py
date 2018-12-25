# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:49:27 2018
@author: GanJinZERO
"""

import numpy as np
import random

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

"""
for i in range(5):
    for j in range(5):
        R[i][j] += 1
"""

alpha = 0.7
lbd = 0.8
epsilon = 0.2
epoch_count = 0
Q_delta = 1024

move_mode = [[1, 0], [0, 1], [-1, 0], [0, -1]]

def draw_answer(state_x, state_y):
    dir_mode = ["↓", "→", "↑", "←"]
    picture = [["R","R","R","R","R"],
               ["R","W","R","W","R"],
               ["R","R","R","R","R"],
               ["R","W","R","W","R"],
               ["R","R","O","R","R"]]
    while R[state_x][state_y] != 100:
        a = np.argmax(Q[state_x][state_y])
        picture[state_x][state_y] = dir_mode[a]
        state_x += move_mode[a][0]
        state_y += move_mode[a][1]
    for line in picture:
        print("".join(line))
    

Q = np.zeros((5, 5, 4))
for i in range(5):
    for j in range(5):
        for k in range(4):
            if not(check_valid_move([i + move_mode[k][0], j + move_mode[k][1]])):
                Q[i][j][k] = -1000
                
Q0 = Q.copy()

while epoch_count < 1000:
    epoch_count += 1
    S_x = random.randint(0, 4)
    S_y = random.randint(0, 4)
    while R[S_x][S_y] != 100:
        u = random.random()
        if u < epsilon:
            a = random.randint(0, 3)
            while not(check_valid_move([S_x + move_mode[a][0], S_y + move_mode[a][1]])):
                a = random.randint(0, 3)
        else:
            Q_now = Q[S_x][S_y]
            a = np.argmax(Q_now)
        S_x1 = S_x + move_mode[a][0]
        S_y1 = S_y + move_mode[a][1]
        Q[S_x][S_y][a] += alpha * (R[S_x1][S_y1] + lbd * max(Q[S_x1][S_y1]) - Q[S_x][S_y][a])
        S_x = S_x1
        S_y = S_y1
    Q_delta = np.sqrt(np.sum(np.square(Q - Q0)))
    Q0 = Q.copy()
    
print(epoch_count)
print(Q_delta)
draw_answer(0, 1)