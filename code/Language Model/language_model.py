# @author: GanjinZero
# @time: 2018/12/25
# @description: language_model.py is a neural network based Japanese language model.

from load_text import load_text
import MeCab
mecab = MeCab.Tagger("-Ochasen")

def mecab_to_text(sentence):
    text_list = []
    m = mecab.parseToNode(sentence)
    while m:
        text_list.append(m.surface)
        m = m.next
    return " ".join(text_list).strip()

japanese_text = load_text()