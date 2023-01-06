#!/usr/bin/env python
# coding:utf-8
import os
import pandas as pd

LANG = 'en'
cat = 'home_and_kitchen'
src_set_num = '1'
dest_set_num = '2'
src_dir = f"/home/junhao.jh/dataset/amazon18/{cat}/set{src_set_num}"
dest_dir = f"/home/junhao.jh/dataset/amazon18/{cat}/set{dest_set_num}"
dest_rel_file = f"{dest_dir}/{cat}.rel.data"
dest_prd_file = f"{dest_dir}/{cat}.prd.data"
dest_rvw_file = f"{dest_dir}/{cat}.rvw.data"
PARA_DIR, FILE = os.path.split(dest_rel_file)


def get_split_data(relf, pf, rf, stage):
    print('Process %s...' % stage)
    src_prd = pd.read_json(os.path.join(src_dir, '%s.prd.%s' % (cat, stage)))
    split_index: set = set(src_prd.product_id.unique())

    def split_func(x):
        return x in split_index

    split_relf = relf[relf.product_id.apply(split_func)]
    split_pf = pf[pf.product_id.apply(split_func)]
    split_rf = rf[rf.product_id.apply(split_func)]

    # save
    split_relf.to_json(os.path.join(PARA_DIR, '%s.rel.%s' % (cat, stage)))
    split_pf.to_json(os.path.join(PARA_DIR, '%s.prd.%s' % (cat, stage)))
    split_rf.to_json(os.path.join(PARA_DIR, '%s.rvw.%s' % (cat, stage)))


if __name__ == '__main__':
    # read whole frame
    relf = pd.read_json(dest_rel_file)
    pf = pd.read_json(dest_prd_file)
    rf = pd.read_json(dest_rvw_file)

    # process data
    get_split_data(relf, pf, rf, 'train')
    get_split_data(relf, pf, rf, 'dev')
    get_split_data(relf, pf, rf, 'test')
