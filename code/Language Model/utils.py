# @author: GanjinZero
# @time: 2018/12/25
# @description: utils.py is some useful functions for Japanese language model.

import os
import time
import numpy as np


def load_text(use_length=-1, min_len=10):
    start = time.clock()
    japanese_text_path = "H:\\Work\\JapaneseModel\\Japanese_book\\"
    text_list = []
    
    if use_length == -1:
        for file in os.listdir(japanese_text_path):
            with open(japanese_text_path + file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line_use = line.strip()
                    if len(line_use) > min_len:
                        text_list.append(line_use)
    else:
        counter = 0
        for file in os.listdir(japanese_text_path):
            with open(japanese_text_path + file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line_use = line.strip()
                    if len(line_use) > min_len:
                        text_list.append(line_use)
                    counter += 1
                    if counter == use_length:
                        print("Japanese text loaded %d lines."%use_length)
                        elapsed = time.clock() - start
                        print("Time used:", round(elapsed, 3))
                        return text_list
                
    print("Japanese text loaded all lines.")
    elapsed = time.clock() - start
    print("Time used:", round(elapsed, 3))

    return text_list

def make_word_dictionary(split_text_list, lower_bound=100):
    start = time.clock()
    
    word_dictionary = dict()
    for sentence in split_text_list:
        sentence_use = sentence.split(" ")
        for word in sentence_use:
            if not word in word_dictionary:
                word_dictionary[word] = 1
            else:
                word_dictionary[word] += 1
                
    print("Word dictionary established.")
    elapsed = time.clock() - start
    print("Time used:", round(elapsed, 3))
    
    if lower_bound > 0:
        pop_list = []
        for word in word_dictionary:
            if word_dictionary[word] < lower_bound:
                pop_list.append(word)
        for word in pop_list:
            word_dictionary.pop(word)
            
    word_list = []
    for word in word_dictionary:
        word_list.append(word)
    
    return word_list

def load_embedding():
    start = time.clock()
    
    """
    Total 2000000 words in this embedding file, 300-d. It is float16 type.
    The first line is "2000000 300".
    You may delete this line.
    """
    EMBEDDING_FILE = 'H:\\Work\\cc.ja.300.vec' 
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float16')
    embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE, 'r', encoding="utf-8"))
    
    elapsed = time.clock() - start
    print("Word vectors loaded.")
    print("Time used:", round(elapsed, 3))
    
    return embeddings_index

def mecab_to_text(sentence_list):
    """
    :param sentence_list: A list of sentences or one single sentence.
    :return: A list of segmented sentences.
    :note: Use mecab to segment a list of sentences or one single sentence in Japanese.
    """
    import MeCab
    mecab = MeCab.Tagger("-Ochasen")

    single_flag = False
    if isinstance(sentence_list, str):
        sentence_list = [sentence_list]
        single_flag = True

    ret_list = []
    for sentence in sentence_list:
        text_list = []
        m = mecab.parseToNode(sentence)
        while m:
            text_list.append(m.surface)
            m = m.next
        seg_sentence = " ".join(text_list).strip()
        ret_list.append(seg_sentence)

    if single_flag:
        return ret_list[0]
    return ret_list

def clear_dictionary(dictionary, embedding_dictionary):
    ret_list = []
    for word in dictionary:
        if word in embedding_dictionary:
            ret_list.append(word)
    return ret_list

def generate_train(window, end_index, text_seq):
    prefix = [0] * (window - 1)
    suffix = [end_index]
    x_list = []
    y_list = []
    for seq in text_seq:
        if len(seq) > 1:
            seq_use = prefix + seq + suffix
            # print(seq_use)
            for i in range(len(seq_use) - window + 1):
                x_list.append(seq_use[i: i + window - 1])
                y_list.append(seq_use[i + window - 1])
                # print(seq_use[i: i + window])
    return x_list, y_list