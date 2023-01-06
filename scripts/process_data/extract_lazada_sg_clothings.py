"""
Processing Lazada dataset
Author: Junhao Liu

2020/11/3
'venture_category1_name_en' is the biggest main category index
First, the program scans the prodcut file and gathers the product id which
belongs to the desired category such as clothing (meta set). After that, the review file
will be scaned and collects the non-null review string which product id is included
in the meta set.

2020/11/18
Found there are multiple product items have same `product_id`. This time, we collect all
the product with `product_id` for further analysing. (After checking the duplicate items,
all of them contain similar info excluding the image url. We have solved this problem in
the `construct product` method. And in this file, we collect all the product with the same
product id for post processing)

2020/12/1
Re-implement the processing and extraction code for the meet of big data file reading and
processing since the product table is very big (>250 GB). The whole processing may consume
about 4 hours.
"""

import os
import datetime

from tqdm import tqdm
from collections import defaultdict
from utils import (
    write_list_to_json,
    write_list_to_file,
    filter_cat_review_length,
    filter_cat_date
)
from cytoolz import partial
from utils import ProcessUnit, get_config_string
from utils_lazada import parse_key, parse_file, match_category, sort_product_key


def extract_cat_metadata(datafile, cat, meta_key):
    print('Extracting product data by cat: %s...' % cat)

    data = defaultdict(list)  # key: prd_id, val: origin str
    build_loader = tqdm(parse_file(datafile, len(meta_key), '!qp!'))
    for i, line in enumerate(build_loader):
        d = line.split('!qp!')

        prd_cat = d[meta_key['venture_category1_name_en']]
        prd_id = d[meta_key['product_id']]
        prd_date = d[meta_key['create_date']]

        if match_category(cat, prd_cat) and \
           filter_cat_date(prd_date, MIN_PRODUCT_DATE, MAX_PRODUCT_DATE):
            data[prd_id].append(line)

    return data, len(data)


def extract_one_cat_metadata(line, target_cat, cat_index, prd_id_index, date_index):
    d = line.split('!qp!')
    prd_cat = d[cat_index]
    prd_id = d[prd_id_index]
    prd_date = d[date_index]

    if match_category(target_cat, prd_cat) and \
       filter_cat_date(prd_date, MIN_PRODUCT_DATE, MAX_PRODUCT_DATE):
        return prd_id, line

    return None


def mp_extract_cat_metadata(process_unit: ProcessUnit, datafile, cat, meta_key):
    """A function to get related product data.

    Args:
        process_unit (ProcessUnit): process unit
        datafile (str): file addr
        cat (list): desired category classification
        meta_key (dict): key file

    Returns:
        Union[set]: return set key data
    """
    print('Extracting product data by cat: %s...' % cat)

    data = set()
    build_loader = tqdm(parse_file(datafile, len(meta_key), '!qp!'))
    cat_index = meta_key['venture_category1_name_en']
    prd_id_index = meta_key['product_id']
    date_index = meta_key['create_date']
    func = partial(
        extract_one_cat_metadata,
        target_cat=cat,
        cat_index=cat_index,
        prd_id_index=prd_id_index,
        date_index=date_index
    )

    for res in process_unit.build_task(func, build_loader):
        prd_id, line = res
        data.add(prd_id)

    return data, len(data)


def filter_one_cat_metadata(line, prd_id_index, filter_prd_id):
    d = line.split('!qp!')
    prd_id = d[prd_id_index]

    if prd_id in filter_prd_id:
        return prd_id, line

    return None


def mp_filter_cat_metadata(process_unit: ProcessUnit, datafile, meta_key, filter_prd_id):
    """A function to get related product data.

    Args:
        process_unit (ProcessUnit): process unit
        datafile (str): file addr
        cat (list): desired category classification
        meta_key (dict): key file
        filter_prd_id (set): the filter

    Returns:
        Union[set, dict]: return set if filter is not given else return dict contains data
    """
    print('Filtering product data by filter_prd_id...')

    filter_data = defaultdict(list)  # key: prd_id, val: origin str
    build_loader = tqdm(parse_file(datafile, len(meta_key), '!qp!'))
    prd_id_index = meta_key['product_id']
    func = partial(
        filter_one_cat_metadata,
        prd_id_index=prd_id_index,
        filter_prd_id=filter_prd_id
    )

    for res in process_unit.build_task(func, build_loader):
        prd_id, line = res
        filter_data[prd_id].append(line)

    return filter_data, len(filter_data)


def extract_cat_review(datafile, review_key, meta_data, meta_key):
    print('Extracting review data...')

    review_num = 0
    hash_review_data = defaultdict(list)  # key: prd_id, val: origin review str
    build_loader = tqdm(parse_file(datafile, len(review_key), '!qp!'))
    for i, line in enumerate(build_loader):
        d = line.split('!qp!')

        # desired category review and filter review
        prd_id = d[review_key['product_id']]
        rvw = d[review_key['review_content']]
        rvw_date = d[review_key['create_date']]

        if prd_id in meta_data and \
           filter_cat_review_length(rvw, MIN_REVIEW_LENGTH, MAX_REVIEW_LENGTH) and \
           filter_cat_date(rvw_date, MIN_REVIEW_DATE, MAX_REVIEW_DATE):
            hash_review_data[prd_id].append(line)
            review_num += 1

    return hash_review_data, review_num


def extract_one_cat_review(line, meta_data, prd_id_index, review_cont_index, date_index):
    d = line.split('!qp!')

    # desired category review and filter review
    prd_id = d[prd_id_index]
    rvw = d[review_cont_index]
    rvw_date = d[date_index]

    if prd_id in meta_data and \
       filter_cat_review_length(rvw, MIN_REVIEW_LENGTH, MAX_REVIEW_LENGTH) and \
       filter_cat_date(rvw_date, MIN_REVIEW_DATE, MAX_REVIEW_DATE):
        return prd_id, line
    
    return None


def mp_extract_cat_review(process_unit: ProcessUnit, datafile, review_key, meta_data, meta_key):
    print('Extracting review data...')

    review_num = 0
    hash_review_data = defaultdict(list)  # key: prd_id, val: origin review str
    build_loader = tqdm(parse_file(datafile, len(review_key), '!qp!'))
    prd_id_index = review_key['product_id']
    review_cont_index = review_key['review_content']
    date_index = review_key['create_date']
    func = partial(
        extract_one_cat_review,
        meta_data=meta_data,
        prd_id_index=prd_id_index,
        review_cont_index=review_cont_index,
        date_index=date_index
    )

    for res in process_unit.build_task(func, build_loader):
        prd_id, line = res
        hash_review_data[prd_id].append(line)
        review_num += 1

    return hash_review_data, review_num


def get_not_null_metadata(product_data: set, review_data: dict):
    '''
    Only save the product with at least one review.
    Sort the pid to ensure all the object are saved alignly.
    '''
    # for pid, ps in product_data.items():
    #     if len(review_data[pid]) > 0:
    #         for p in ps:
    #             yield p
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


def align_metadata(product_data: set, review_data: dict):
    '''
    Align review to the product item
    '''
    print('Align metadata and review_data...')

    for pid in tqdm(get_not_null_metadata(product_data, review_data)):
        # yield {'product': product_data[pid], 'review': review_data[pid]}
        yield {'product_id': pid, 'review': review_data[pid]}


if __name__ == '__main__':
    '''
    Setting
    '''
    max_core = 1
    cat = 'clothing_prd_y18_19'
    filter_cat = ['cloth', 'shoe', 'jewellery', 'jewelery', 'fashion']
    # filter_cat = ['electronic', 'mobile', 'tablet', 'camera', 'smart device', 'laptop', 'computer']

    MAX_REVIEW_LENGTH = None
    MIN_REVIEW_LENGTH = 0
    MAX_REVIEW_DATE = datetime.date(2020, 7, 31)
    MIN_REVIEW_DATE = datetime.date(2017, 12, 31)
    MAX_PRODUCT_DATE = datetime.date(2020, 1, 1)
    MIN_PRODUCT_DATE = datetime.date(2017, 12, 31)

    '''
    Server
    '''
    key_file = "scripts/process_data/lazada_key.txt"
    meta_file = [
        "/home/zhen.hai/Project/MAP/original_data_tables/table_lzd_prd_sku_core/haiz_dim_lzd_prd_sku_core_sg_y2018.txt",
        "/home/zhen.hai/Project/MAP/original_data_tables/table_lzd_prd_sku_core/haiz_dim_lzd_prd_sku_core_sg_y2019.txt"
    ]
    review_file = [
        "/home/zhen.hai/Project/MAP/original_data_tables/table_lzd_rvw_prd_df/haiz_dwd_lzd_rvw_prd_sg_y20y19y18.txt"
    ]
    dest_dir = "/home/junhao.jh/dataset/lazada_en/%s" % cat
    dest_metadata_id_file = "%s/%s.product_id" % (dest_dir, cat)
    dest_metadata_file = "%s/%s.product" % (dest_dir, cat)
    dest_review_file = "%s/%s.review" % (dest_dir, cat)
    dest_review_origin_file = "%s/%s.review_origin" % (dest_dir, cat)
    dest_cat_file = "%s/%s.dataset_config" % (dest_dir, cat)

    '''
    Local (for debug)
    '''
    # key_file = "scripts/process_data/lazada_key.txt"
    # review_file = "dataset/origin/lazada/rvw_sample.txt"
    # meta_file = "dataset/origin/lazada/prd_sample.txt"
    # dest_dir = "dataset/origin/lazada/%s" % cat
    # dest_metadata_file = "%s/%s.metadata.data" % (dest_dir, cat)
    # dest_review_file = "%s/%s.review.data" % (dest_dir, cat)
    # dest_cat_file = "%s/%s.cat.data" % (dest_dir, cat)
    # dest_align_file = "%s/%s.align.data" % (dest_dir, cat)

    '''
    Start extract
    '''
    print('dir:', dest_dir)
    print(key_file)
    print(review_file)
    print(meta_file)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # get process unit
    process_unit = ProcessUnit(max_core)

    # write category and config
    save_config = [
        "filter_cat",
        "meta_file",
        "review_file",
        "MAX_REVIEW_LENGTH",
        "MIN_REVIEW_LENGTH",
        "MAX_REVIEW_DATE",
        "MIN_REVIEW_DATE",
        "MAX_PRODUCT_DATE",
        "MIN_PRODUCT_DATE"
    ]
    write_list_to_file(
        sum([[i + ':', eval('str(%s)' % i), ''] for i in save_config], []),
        dest_cat_file,
        join=False
    )

    # parse key
    prd_key, rvw_key = parse_key(key_file)

    # get meta product id key
    meta_data, prd_num = mp_extract_cat_metadata(process_unit, meta_file, filter_cat, prd_key)
    print('Found product: %d' % prd_num)

    # get review data
    review_data, rvw_num = mp_extract_cat_review(process_unit, review_file, rvw_key, meta_data, prd_key)
    print('Found review: %d' % rvw_num)

    # stat, align and save review data
    align_generator = align_metadata(meta_data, review_data)
    align_num = write_list_to_json(align_generator, dest_review_file)
    prd_num = write_list_to_file(
        get_not_null_metadata(meta_data, review_data),
        dest_metadata_id_file,
        join=False)
    rvw_num = write_list_to_file(
        get_flatten_reviewdata(review_data),
        dest_review_origin_file,
        join=False)
    print('Aligned product: %d, review: %d' % (prd_num, rvw_num))

    # get and save meta product data
    filter_prd_id_set = set(get_not_null_metadata(meta_data, review_data))
    meta_data.clear()  # release meta
    review_data.clear()  # release review data memory
    filter_meta_data, filter_num = mp_filter_cat_metadata(process_unit, meta_file, prd_key, filter_prd_id_set)
    write_list_to_json(
        ({'product_id': k, 'product': filter_meta_data[k]} for k in sort_product_key(filter_prd_id_set)),
        dest_metadata_file)
