import os

import pandas as pd
import matchzoo
from cytoolz import partial


def _read_data(prd_path,
               rvw_path,
               rel_path,
               task):
    prd_table = pd.read_json(prd_path)
    rvw_table = pd.read_json(rvw_path)
    rel_table = pd.read_json(rel_path)

    prd = pd.DataFrame({
        'text_left': prd_table['name'],
        'id_left': prd_table['product_id'],
    })
    prd.id_left = prd.id_left.astype(str)
    prd = prd.reset_index(drop=True)

    rvw = pd.DataFrame({
        'text_right': rvw_table['content'],
        'id_right': rvw_table['review_id']
    })
    rvw.id_right = rvw.id_right.astype(str)
    rvw = rvw.reset_index(drop=True)

    rel_table['upvotes'] = rel_table['upvotes'].astype(int)
    rel_table.loc[rel_table['upvotes'] > 1, 'upvotes'] = 2
    rel_table.loc[rel_table['upvotes'] < 0, 'upvotes'] = 0

    rel = pd.DataFrame({
        'id_left': rel_table['product_id'],
        'id_right': rel_table['review_id'],
        'label': rel_table['upvotes']
    })
    rel.id_left = rel.id_left.astype(str)
    rel = rel.reset_index(drop=True)

    return matchzoo.map_pack(prd, rvw, relation=rel, task=task)


def upvotes_map_func(upvote):
    if upvote >= 16:
        return 4
    
    if upvote >= 8:
        return 3
    
    if upvote >= 4:
        return 2
    
    if upvote >= 2:
        return 1

    if upvote == 1:
        return 0
    
    return -1


def _read_data2(prd_path,
                rvw_path,
                rel_path,
                task,
                features_path):
    """Read the json file with multiple intervel training.

    And drop review without upvotes.
    """
    prd_table = pd.read_json(prd_path)
    rvw_table = pd.read_json(rvw_path)
    rel_table = pd.read_json(rel_path)

    prd = pd.DataFrame({
        'text_left': prd_table['name'],
        'id_left': prd_table['product_id'],
    })
    prd.id_left = prd.id_left.astype(str)
    prd = prd.reset_index(drop=True)

    rvw = pd.DataFrame({
        'text_right': rvw_table['content'],
        'id_right': rvw_table['review_id']
    })
    rvw.id_right = rvw.id_right.astype(str)
    rvw = rvw.reset_index(drop=True)

    rel_table['upvotes'] = rel_table['upvotes'].astype(int)

    # mapping upvotes number into different categories
    rel_table.upvotes = rel_table.upvotes.apply(upvotes_map_func)

    rel: pd.DataFrame = pd.DataFrame({
        'id_left': rel_table['product_id'],
        'id_right': rel_table['review_id'],
        'label': rel_table['upvotes']
    })
    rel.id_left = rel.id_left.astype(str)
    rel = rel.reset_index(drop=True)

    # remove data with 0 upvote (label == -1) as noise
    zero_upvote_index = rel[rel.label == -1].index
    rel.drop(index=zero_upvote_index, inplace=True)

    return matchzoo.map_pack(prd, rvw, relation=rel, task=task)


def _read_data3(prd_path,
                rvw_path,
                rel_path,
                task,
                features_path):
    """Read the json file with multiple intervel training.

    Similar to _read_data2 and this version add the product description to text_left.
    """
    prd_table = pd.read_json(prd_path)
    rvw_table = pd.read_json(rvw_path)
    rel_table = pd.read_json(rel_path)

    prd = pd.DataFrame({
        'text_left': list(map(lambda x: f"{x[0]} {x[1]}", zip(prd_table['name'].tolist(), prd_table['description'].tolist()))),
        'id_left': prd_table['product_id'],
    })
    prd.id_left = prd.id_left.astype(str)
    prd = prd.reset_index(drop=True)

    rvw = pd.DataFrame({
        'text_right': rvw_table['content'],
        'id_right': rvw_table['review_id']
    })
    rvw.id_right = rvw.id_right.astype(str)
    rvw = rvw.reset_index(drop=True)

    rel_table['upvotes'] = rel_table['upvotes'].astype(int)

    # mapping upvotes number into different categories
    rel_table.upvotes = rel_table.upvotes.apply(upvotes_map_func)

    rel: pd.DataFrame = pd.DataFrame({
        'id_left': rel_table['product_id'],
        'id_right': rel_table['review_id'],
        'label': rel_table['upvotes']
    })
    rel.id_left = rel.id_left.astype(str)
    rel = rel.reset_index(drop=True)

    # remove data with 0 upvote (label == -1) as noise
    zero_upvote_index = rel[rel.label == -1].index
    rel.drop(index=zero_upvote_index, inplace=True)

    return matchzoo.map_pack(prd, rvw, relation=rel, task=task)


def img_exist(dir_name, features_path, split):
    files_dir = os.path.join(
        f"{features_path}/{split}", dir_name)
    pth_path = os.path.join(files_dir, f"{dir_name}.pth")
    if os.path.exists(pth_path):
        return 1
    return 0


def _read_data4(prd_path,
                rvw_path,
                rel_path,
                task,
                features_path):
    """Read the json file with multiple intervel training.

    Similar to _read_data3 and this version only consider the review with image.
    """
    prd_table = pd.read_json(prd_path)
    rvw_table = pd.read_json(rvw_path)
    rel_table = pd.read_json(rel_path)
    
    # features_path = "/home/junhao.jh/dataset/lazada/home_prd_y18_19/set2/features"
    split = str(prd_path).split('.')[-1]

    prd = pd.DataFrame({
        'text_left': list(map(lambda x: f"{x[0]} {x[1]}", zip(prd_table['name'].tolist(), prd_table['description'].tolist()))),
        'id_left': prd_table['product_id'],
    })
    prd.id_left = prd.id_left.astype(str)
    prd = prd.reset_index(drop=True)

    rvw = pd.DataFrame({
        'text_right': rvw_table['content'],
        'id_right': rvw_table['review_id']
    })
    rvw.id_right = rvw.id_right.astype(str)
    rvw = rvw.reset_index(drop=True)

    rel_table['upvotes'] = rel_table['upvotes'].astype(int)

    # mapping upvotes number into different categories
    rel_table.upvotes = rel_table.upvotes.apply(upvotes_map_func)

    rel: pd.DataFrame = pd.DataFrame({
        'id_left': rel_table['product_id'],
        'id_right': rel_table['review_id'],
        'label': rel_table['upvotes']
    })
    rel.id_left = rel.id_left.astype(str)
    rel = rel.reset_index(drop=True)

    # remove data with 0 upvote (label == -1) as noise
    zero_upvote_index = rel[rel.label == -1].index
    rel.drop(index=zero_upvote_index, inplace=True)

    # remove review data without image
    wo_image_index = rel[rel.id_right.apply(partial(img_exist, features_path=features_path, split=split)) == 0].index
    rel.drop(index=wo_image_index, inplace=True)

    return matchzoo.map_pack(prd, rvw, relation=rel, task=task)
