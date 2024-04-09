# coding:utf-8
import sys
import os
import json
import re
import numpy as np

def load_data(data_path):
    """
    return word2ix: dict,每个字对应的序号，形如u'月'->100
    return ix2word: dict,每个序号对应的字，形如'100'->u'月'
    return poet_data: numpy数组，每一行是一首诗对应的字的下标
    """
    if os.path.isfile(data_path):
        data = np.load(data_path, allow_pickle=True)
        poet_data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()

        return word2ix, ix2word, poet_data
    else:
        print('[ERROR] Data File Not Exists')
        exit()


def decode_poetry(idx, word2ix, ix2word, poet_data):
    """
    解码诗歌数据
    输入:
        idx: 第几首诗歌(共311823首，idx in [0, 311822])
    """
    assert (idx < poet_data.shape[0] and idx >= 0)

    row = poet_data[idx]

    results = ''.join([
        ix2word[char] if ix2word[char] != '</s>' and ix2word[char] != '<EOP>'
                         and ix2word[char] != '<START>' else ''
        for char in row
    ])
    return results

word2ix, ix2word, poet_data=load_data(r'D:\A1Python\人工智能与Python\作业\大作业（4）\古诗生成作业_problem\data\Poetry_data_word2ix_ix2word.npz')