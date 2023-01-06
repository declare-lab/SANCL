import os
from multiprocessing import Pool
from tqdm import tqdm
from utils_lazada import parse_key, parse_file
from utils import write_list_to_json, write_list_to_file
from collections import Counter
from functools import reduce
from cytoolz import partial


def process_one_line(line, cat_index, prd_index):
    d = line.split('!qp!')
    return d[cat_index], d[prd_index]

    
def mp_stat_cat_metadata(datafile, meta_key):
    cat_data = Counter()
    prd_id_pool = set()
    func = partial(
        process_one_line,
        cat_index=meta_key['venture_category1_name_en'],
        prd_index=meta_key['product_id']
    )
    data_iter = tqdm(parse_file(datafile, len(meta_key), '!qp!'))
    
    with Pool(max_core) as pool:
        res = pool.imap_unordered(func, data_iter, chunksize=1000000)
        for c, prd_id in res:
            cat_data[c] += 1
            prd_id_pool.add(prd_id)
    
    return cat_data, prd_id_pool


def stat_cat_metadata(datafile, meta_key):
    cat_data = Counter()
    prd_id_pool = set()

    for i, line in enumerate(tqdm(parse_file(datafile, len(meta_key), '!qp!'))):
        d = line.split('!qp!')
        # print('----')
        # print(d[meta_key['parent_category_name_en']], d[meta_key['parent_category_name_l10n']], sep=' || ')
        # print(d[meta_key['venture_category1_name_en']], d[meta_key['venture_category1_name_l10n']], sep=' || ')
        # print(d[meta_key['venture_category_name_en']], d[meta_key['venture_category_name_l10n']], sep=' || ')
        
        c = d[meta_key['venture_category1_name_en']]
        prd_id = d[meta_key['product_id']]

        cat_data[c] += 1
        prd_id_pool.add(prd_id)

    return cat_data, prd_id_pool


def stat_duplicate_prd(prd_id_pool: dict):
    assert len(prd_id_pool) > 1

    # conjunction
    prd_conj = reduce(lambda x, y: x & y, prd_id_pool.values())
    if len(prd_conj) == 0:
        print("Dont have the same prd id")
    else:
        print("Have the same prd id number:")
        print(len(prd_conj))


if __name__ == '__main__':
    top_k = 500
    max_core = 2

    key_file = "scripts/process_data/lazada_key.txt"
    src_dir = "/home/zhen.hai/Project/MAP/original_data_tables/table_lzd_prd_sku_core"
    meta_files = {
        "y18": f"{src_dir}/haiz_dim_lzd_prd_sku_core_sg_y2018.txt",
        "y19": f"{src_dir}/haiz_dim_lzd_prd_sku_core_sg_y2019.txt"
    }
    dest_dir = "/home/junhao.jh/dataset/lazada/"
    dest_metadata_file = f"{dest_dir}/all.metadata.cat_stat"
    dest_most_meta_data_file = f'{dest_dir}/all.metadata.most_common_{top_k}_cat_stat'

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    print(key_file)
    print(meta_files)
    prd_id_pool = {}

    for key, meta_file in meta_files.items():
        print("Processing %s..." % key)
        print(meta_file)

        # extract category data
        prd_key, rvw_key = parse_key(key_file)
        origin_meta_data, prd_ids = stat_cat_metadata(meta_file, prd_key)
        most_common_meta_data = origin_meta_data.most_common(top_k)

        # save results
        print(most_common_meta_data)
        write_list_to_file(most_common_meta_data, dest_most_meta_data_file + f'_{key}', join_token='\t')
        write_list_to_json([dict(origin_meta_data)], dest_metadata_file + f'_{key}')

        # to find whether duplicate prd id
        prd_id_pool[key] = prd_ids
    
    stat_duplicate_prd(prd_id_pool)
