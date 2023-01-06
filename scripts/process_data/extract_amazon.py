"""
Processing Amazon dataset
Author: Junhao Liu

2020/12/24
"""

import os
import json

from tqdm import tqdm
from collections import defaultdict
from utils import (
    write_list_to_json,
    write_list_to_file
)
from utils_amazon import parse_gzip_file
from utils_lazada import sort_product_key


def extract_cat_metadata(datafile):
    data = defaultdict(list)  # key: prd_id, val: origin str
    build_loader = tqdm(parse_gzip_file(datafile))
    for i, line in enumerate(build_loader):
        asin = line['asin']
        data[asin].append(json.dumps(line))
    return data, len(data)


def extract_cat_review(datafile):
    print('Extracting review data...')

    review_num = 0
    hash_review_data = defaultdict(list)  # key: prd_id, val: origin review str
    build_loader = tqdm(parse_gzip_file(datafile))
    for i, line in enumerate(build_loader):
        votes = int(line.get('vote', '0').replace(',', '').replace(' ', ''))
        if votes > 0:
            asin = line['asin']
            hash_review_data[asin].append(json.dumps(line))
            review_num += 1
    return hash_review_data, review_num


def get_not_null_metadata(product_data: set, review_data: dict):
    '''
    Only save the product with at least one review.
    Sort the pid to ensure all the object are saved alignly.
    '''
    for pid in sort_product_key(product_data):
        if pid in review_data and len(review_data[pid]) > 0:
            yield pid


def get_flatten_reviewdata(review_data: dict):
    '''
    Save the review data in list line by line
    '''
    for _, rs in review_data.items():
        for r in rs:
            yield r


def align_metadata(product_data: dict, review_data: dict):
    '''
    Align review to the product item
    '''
    print('Align metadata and review_data...')

    for pid in tqdm(get_not_null_metadata(set(product_data.keys()), review_data)):
        yield {'product_id': pid, 'review': review_data[pid]}


if __name__ == '__main__':
    '''
    Setting
    '''
    max_core = 1
    cat = 'home_and_kitchen'

    '''
    Server
    '''
    meta_file = [
        "/home/wh.347426/dataset/amazon18/origin/meta_Home_and_Kitchen.json.gz"
    ]
    review_file = [
        "/home/wh.347426/dataset/amazon18/origin/Home_and_Kitchen.json.gz"
    ]
    dest_dir = "/home/wh.347426/dataset/amazon18/%s" % cat
    dest_metadata_id_file = "%s/%s.product_id" % (dest_dir, cat)
    dest_metadata_file = "%s/%s.product" % (dest_dir, cat)
    dest_review_file = "%s/%s.review" % (dest_dir, cat)
    dest_review_origin_file = "%s/%s.review_origin" % (dest_dir, cat)

    '''
    Start extract
    '''
    print('dir:', dest_dir)
    print(review_file)
    print(meta_file)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # get meta product id key
    meta_data, prd_num = extract_cat_metadata(meta_file)
    print('Found product: %d' % prd_num)

    # get review data
    review_data, rvw_num = extract_cat_review(review_file)
    print('Found review: %d' % rvw_num)

    # stat, align and save review data
    align_generator = align_metadata(meta_data, review_data)
    align_num = write_list_to_json(align_generator, dest_review_file)
    prd_num = write_list_to_file(
        get_not_null_metadata(set(meta_data.keys()), review_data),
        dest_metadata_id_file,
        join=False)
    rvw_num = write_list_to_file(
        get_flatten_reviewdata(review_data),
        dest_review_origin_file,
        join=False)
    print('Aligned product: %d, review: %d' % (prd_num, rvw_num))

    # get and save meta product data
    filter_prd_id_set = set(get_not_null_metadata(set(meta_data.keys()), review_data))
    review_data.clear()  # release review data memory
    write_list_to_json(
        ({'product_id': k, 'product': meta_data[k]} for k in sort_product_key(filter_prd_id_set)),
        dest_metadata_file)
