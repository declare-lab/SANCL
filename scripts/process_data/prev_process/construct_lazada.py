# Processing Lazada dataset
# 2020/11/3
#
# venture_category1_name_en is the biggest main category
#
import json
import pandas

from tqdm import tqdm
from collections import defaultdict
from extract_lazada import parse_key


def parse_aligned(input_file, product_key, review_key):
    with open(input_file, "r") as reader:
        for line in reader:
            data = json.loads(line)
            product = data['product']
            reviews = data['review']

            product_return = {k: v for k, v in zip(
                product_key, product.split('!qp!'))}
            reviews_return = []
            for review in reviews:
                reviews_return.append(
                    {k: v for k, v in zip(review_key, review.split('!qp!'))})

            yield product_return, reviews_return


def parse_seperated_aligned(product_file, review_file, product_key, review_key):
    with open(product_file, "r") as prd_reader, open(review_file, "r") as rvw_reader:
        for prd_line, rvw_line in zip(prd_reader, rvw_reader):
            product_info = json.loads(prd_line)
            reviews_info = json.loads(rvw_line)
            assert product_info['product_id'] == reviews_info['product_id']

            product = product_info['product']
            reviews = reviews_info['review']
            product_return = {k: v for k, v in zip(
                product_key, product.split('!qp!'))}
            reviews_return = []
            for review in reviews:
                reviews_return.append(
                    {k: v for k, v in zip(review_key, review.split('!qp!'))})

            yield product_return, reviews_return


def have_upvote(reviews, threshold):
    for review in reviews:
        if int(review['upvotes']) >= threshold:
            return True
    return False


def process_data(data_file,
                 product_key: list,
                 review_key: list,
                 threshold: int,
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

    for product, reviews in tqdm(parse_aligned(data_file, product_key, review_key)):
        prodcut_id = str(product['product_id'])

        # have positive review?
        if not have_upvote(reviews, threshold):
            continue

        # build product frame
        product_frame['product_id'].append(prodcut_id)
        for k, v in extract_product_key.items():
            product_frame[k].append(product[v])

        # build review frame
        upvotes = []
        rating = []
        for idx, review in enumerate(reviews):
            reviews_frame['product_id'].append(prodcut_id)
            reviews_frame['review_id'].append('%s-%d' % (prodcut_id, idx))
            for k, v in extract_review_key.items():
                reviews_frame[k].append(review[v])
            upvotes.append(float(review['upvotes']))
            rating.append(float(review['rating']))

        # build relation frame
        for idx in range(len(reviews)):
            relation_frame['product_id'].append(prodcut_id)
            relation_frame['review_id'].append('%s-%d' % (prodcut_id, idx))
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
    Setting
    '''
    cat = 'clothing'
    threshold = 2
    '''
    Server
    '''
    key_file = "scripts/process_data/lazada_key.txt"
    align_file = f"/home/junhao.jh/dataset/lazada/{cat}/{cat}.align.data"
    dest_rel_file = f"/home/junhao.jh/dataset/lazada/{cat}/{cat}.rel.data"
    dest_prd_file = f"/home/junhao.jh/dataset/lazada/{cat}/{cat}.prd.data"
    dest_rvw_file = f"/home/junhao.jh/dataset/lazada/{cat}/{cat}.rvw.data"

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
    print(key_file)
    print(align_file)
    print(dest_rel_file)
    print(dest_prd_file)
    print(dest_rvw_file)

    '''
    Process for the task specific details
    '''
    # parse key
    prd_key, rvw_key = parse_key(key_file, return_dict=False)

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

    # process review and meta data
    relf, pf, rf = process_data(align_file, prd_key, rvw_key, threshold,
                                extract_prd_key,
                                extract_rvw_key)

    relf.to_json(dest_rel_file)
    pf.to_json(dest_prd_file)
    rf.to_json(dest_rvw_file)
