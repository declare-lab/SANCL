#!/usr/bin/env python
# coding:utf-8
import os
import numpy as np
import pandas as pd

from icu_tokenizer import Tokenizer
from sklearn.model_selection import train_test_split


"""
WoS Reference: https://github.com/kk7nc/HDLTex
"""

LANG = 'en'
cat = 'home_and_kitchen'
set_num = 1

dest_dir = f"/home/wh.347426/dataset/amazon18/{cat}/set{set_num}"
dest_rel_file = f"{dest_dir}/{cat}.rel.data"
dest_prd_file = f"{dest_dir}/{cat}.prd.data"
dest_rvw_file = f"{dest_dir}/{cat}.rvw.data"
PARA_DIR, FILE = os.path.split(dest_rel_file)
total_len = []
np.random.seed(7)

# Create a Tokenizer
tokenizer = Tokenizer(lang=LANG)


def tokenize_sentence(sent):
    sent = tokenizer(sent)
    return list([w.text for w in sent])


def get_split_data(relf, pf, rf, stage, split_index: set):
    print('Process %s...' % stage)

    def split_func(x):
        return x in split_index

    split_relf = relf[relf.product_id.apply(split_func)]
    split_pf = pf[pf.product_id.apply(split_func)]
    split_rf = rf[rf.product_id.apply(split_func)]

    # save
    split_relf.to_json(os.path.join(PARA_DIR, '%s.rel.%s' % (cat, stage)))
    split_pf.to_json(os.path.join(PARA_DIR, '%s.prd.%s' % (cat, stage)))
    split_rf.to_json(os.path.join(PARA_DIR, '%s.rvw.%s' % (cat, stage)))


def split_train_dev_test():
    prd_data = pd.read_json(dest_prd_file)
    prd_id = prd_data.product_id.unique()

    train, test = train_test_split(prd_id, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)

    train = set(train)
    val = set(val)
    test = set(test)

    print(len(train), len(val), len(test))
    return train, val, test


if __name__ == '__main__':
    # split by product item
    print('Split data...')
    train, dev, test = split_train_dev_test()

    # read whole frame
    relf = pd.read_json(dest_rel_file)
    pf = pd.read_json(dest_prd_file)
    rf = pd.read_json(dest_rvw_file)

    # process data
    get_split_data(relf, pf, rf, 'train', train)
    get_split_data(relf, pf, rf, 'dev', dev)
    get_split_data(relf, pf, rf, 'test', test)
