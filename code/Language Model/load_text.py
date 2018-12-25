# @author: GanjinZero
# @time: 2018/12/25
# @description: load_text.py is disigned to load Japanese text book.

import os
import time


def load_text():
    start = time.clock()
    
    japanese_text_path = "H:\\Work\\JapaneseModel\\Japanese_book\\"
    text_list = []

    for file in os.listdir(japanese_text_path):
        with open(japanese_text_path + file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line_use = line.strip()
                text_list.append(line_use)
                
    print("Japanese text loaded.")
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

    return text_list

def generate_train(window=5, maxlen=100000):
    return 0