"""
Processing Lazada dataset
Author: Junhao Liu

2020/11/3
venture_category1_name_en is the biggest main category

"""
import os
import json
import pandas

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from utils_lazada import parse_aligned


# def find_prd_diff(data_file,
#                   product_key: list,
#                   review_key: list,):

#     diff_key = set()
#     for product, reviews in tqdm(parse_aligned(data_file, product_key, review_key)):
#         for k in product[0]:
#             diff = set()
#             for prd in product:
#                 diff.add(prd[k])
#             if len(diff) > 1:
#                 diff_key.add(k)
    
#     return diff_key


def merge_products(product_list: list, merge_key: dict):
    """Merge multiple products with the same product id into a single representation,
    has considered the different product may contain the same value.

    Args:
        product_list (list): the products with the same product index
        merge_key (dict): the products key need to be merged

    Returns:
        dict: the merged dict of product
    """
    product = deepcopy(product_list[0])

    for mk in merge_key:
        whole = set()
        whole_key = merge_key[mk]
        for prd in product_list:
            for k in whole_key:
                v = prd[k]
                if len(v) > 0:
                    whole.add(v)
        
        if len(whole) > 0:
            save_v = json.dumps(list(whole))
            product[mk] = save_v
    return product


def process_data(data_file,
                 merge_key: dict,
                 extract_product_key: dict,
                 extract_review_key: dict):
    """
    Extract the aligned meta-review file into the data structure
    with ranking support file.

    Note:
    1. For those prodcuts w/o upvotes >= threshold review will be filtered.
    2. Return three dataframes which is compact with the matchzoo libarary
    """
    product_frame = defaultdict(list)
    reviews_frame = defaultdict(list)
    relation_frame = defaultdict(list)

    for products, reviews in tqdm(parse_aligned(data_file)):
        product_id = str(products[0]['product_id'])

        # merge mutltiple products
        product = merge_products(products, merge_key)

        # build product frame
        product_frame['product_id'].append(product_id)
        for k, v in extract_product_key.items():
            product_frame[k].append(product[v])

        # build review frame
        upvotes = []
        rating = []
        for idx, review in enumerate(reviews):
            reviews_frame['product_id'].append(product_id)
            reviews_frame['review_id'].append('%s-%d' % (product_id, idx))
            for k, v in extract_review_key.items():
                reviews_frame[k].append(review[v])
            upvotes.append(float(review['upvotes']))
            rating.append(float(review['rating']))

        # build relation frame
        for idx in range(len(reviews)):
            relation_frame['product_id'].append(product_id)
            relation_frame['review_id'].append('%s-%d' % (product_id, idx))
        relation_frame['upvotes'].extend(upvotes)
        relation_frame['rating'].extend(rating)

    # build
    pf = pandas.DataFrame(product_frame)
    rf = pandas.DataFrame(reviews_frame)
    relf = pandas.DataFrame(relation_frame)
    relf.product_id = relf.product_id.astype(str)
    relf.review_id = relf.review_id.astype(str)

    print(pf.head())
    print(rf.head())
    print(relf.head())

    # print info
    print('Product num:', len(pf))
    print('Review num:', len(rf))
    print('Pair num:', len(relf))
    return (relf, pf, rf)


if __name__ == '__main__':
    '''
    Server
    '''
    align_file = "/home/junhao.jh/dataset/lazada/home_prd_y18_19/set2/home_prd_y18_19.align"
    dest_dir, cat = os.path.split(align_file)
    cat = cat.split('.')[0]
    dest_rel_file = f"{dest_dir}/{cat}.rel.data"
    dest_prd_file = f"{dest_dir}/{cat}.prd.data"
    dest_rvw_file = f"{dest_dir}/{cat}.rvw.data"

    '''
    Local
    '''
    # key_file = "scripts/process_data/lazada_key.txt"
    # align_file = "dataset/origin/lazada/%s.align.data" % cat
    # dest_rel_file = "dataset/origin/lazada/%s.rel.data" % cat
    # dest_prd_file = "dataset/origin/lazada/%s.prd.data" % cat
    # dest_rvw_file = "dataset/origin/lazada/%s.rvw.data" % cat

    '''
    Start process
    '''
    print(align_file)
    print(dest_rel_file)
    print(dest_prd_file)
    print(dest_rvw_file)

    '''
    Process for the task specific details
    '''
    # merge product key
    merge_prd_key = {
        "url_main_image": [
            "url_main_image",
            "image_url_2",
            "image_url_3",
            "image_url_4",
            "image_url_5"
        ],
        "description": [
            "description"
        ]
    }

    # defind field
    extract_prd_key = {
        "name": "product_name",
        "img_url": "url_main_image",
        "keyword": "keywords",
        "description": "description",
        "brand_name": "brand_name"
    }

    extract_rvw_key = {
        "title": "review_title",
        "content": "review_content",
        "img_url": "images"
    }

    # find product key
    # diff = find_prd_diff(align_file, prd_key, rvw_key)
    # print(diff)

    # process review and meta data
    relf, pf, rf = process_data(align_file,
                                merge_prd_key,
                                extract_prd_key,
                                extract_rvw_key)

    relf.to_json(dest_rel_file)
    pf.to_json(dest_prd_file)
    rf.to_json(dest_rvw_file)
