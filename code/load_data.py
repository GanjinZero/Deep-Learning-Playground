# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:18:43 2018

@author: GanJinZERO
"""

import os
from PIL import Image
import numpy as np

def load_data_anime_face(nb=51222):
    train_set = []
    file_path = "../data/anime_face/"
    counter = 0

    for picture_path in os.listdir(file_path):
        im = Image.open(file_path + picture_path)
        im_array = np.array(im)
        train_set.append(im_array)
        counter += 1
        if counter == nb:
            break
    
    return train_set