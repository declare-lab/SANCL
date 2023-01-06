"""
Clean and filter the lazada dataset
Author: Junhao Liu

Summary:
    1) filter too short review
    2) replace emoji with text
    3) filter products with only a few of reviews
    4) filter products with too short title name (optional)
    5) filter products without positive review (by setting threshold)

2020/12/1
Some review contain emoji.

2020/12/2
Add filtering strategy
"""
import os
import json

from tqdm import tqdm
from cytoolz import partial
from utils import (
    filter_cat_review_length,
    write_list_to_file,
    write_list_to_json
)
from utils_amazon import(
    parse_separated_aligned,
    have_upvote,
    parse_url
)


class CleanPipeline:
    def __init__(self):
        self.prd_pipeline = []
        self.rvw_pipeline = []

    def __call__(self, prd_list: list, rvw_list: list):
        prd_list, rvw_list = self.run_rvw_pipeline(prd_list, rvw_list)
        prd_list, rvw_list = self.run_prd_pipeline(prd_list, rvw_list)
        return prd_list, rvw_list

    # def run_rvw_pipeline(self, prd_list: list, rvw_list: list):
    #     # review processing pipeline
    #     filter_rvw = []
    #     for rvw_dict in rvw_list:
    #         rvw = rvw_dict['review_content']
    #         for p in self.rvw_pipeline:
    #             res = p(rvw)
    #             if isinstance(res, str):  # return string
    #                 rvw = res
    #             else:  # return bool
    #                 if not res:
    #                     rvw = None
    #                     break
    #         if rvw:
    #             rvw_dict['review_content'] = rvw
    #             filter_rvw.append(rvw_dict)
    #     return prd_list, filter_rvw

    def _run_prd_pipeline(self,
                          prd_list: list,
                          rvw_list: list,
                          min_rvw_num,
                          min_rvw_positive):
        # product processing pipeline
        if not rvw_list:
            return None, None

        if len(rvw_list) < min_rvw_num:
            return None, None

        if not have_upvote(rvw_list, min_rvw_positive):
            return None, None

        return prd_list, rvw_list

    def _run_rvw_pipeline(self,
                          prd_list: list,
                          rvw_list: list,
                          min_rvw_len,
                          max_rvw_len,
                          force_rvw_img):
        # review processing pipeline
        filter_rvw = []
        for rvw_dict in rvw_list:
            rvw_ok = True
            rvw = rvw_dict.get('reviewText', "")
            rvw_img = rvw_dict.get('image', [])

            if force_rvw_img and len(rvw_img) == 0:
                rvw_ok = False

            if not filter_cat_review_length(rvw, min_length=min_rvw_len, max_length=max_rvw_len):
                rvw_ok = False

            if rvw_ok:
                rvw_dict['reviewText'] = rvw
                filter_rvw.append(rvw_dict)
        return prd_list, filter_rvw

    def build_rvw_pipeline(self, min_rvw_len, max_rvw_len, force_rvw_img):
        setattr(self, 'run_rvw_pipeline', partial(
            self._run_rvw_pipeline,
            min_rvw_len=min_rvw_len,
            max_rvw_len=max_rvw_len,
            force_rvw_img=force_rvw_img
        ))

    def build_prd_pipeline(self, min_rvw_num, min_rvw_positive):
        setattr(self, 'run_prd_pipeline', partial(
            self._run_prd_pipeline,
            min_rvw_num=min_rvw_num,
            min_rvw_positive=min_rvw_positive
        ))


def clean_data(line_num=None):
    cleaner = CleanPipeline()
    cleaner.build_prd_pipeline(min_rvw_num, min_rvw_positive)
    cleaner.build_rvw_pipeline(min_rvw_len, max_rvw_len, force_rvw_img)

    for line, outputs in enumerate(tqdm(parse_separated_aligned(prd_file, rvw_file))):
        product_id, products, reviews = outputs
        prd_res, rvw_res = cleaner(products, reviews)

        if prd_res and rvw_res:
            yield {
                'product_id': product_id,
                'product': prd_res,
                'review': rvw_res
            }
        if line_num and line > line_num:
            break


if __name__ == '__main__':
    # setting
    clean_name = 'set1'
    cat = "home_and_kitchen"
    prd_file = f"/home/wh.347426/dataset/amazon18/{cat}/{cat}.product"
    rvw_file = f"/home/wh.347426/dataset/amazon18/{cat}/{cat}.review"
    min_rvw_len = 64
    max_rvw_len = None
    min_rvw_num = 20
    max_rvw_num = None
    min_rvw_positive = 2
    force_rvw_img = False

    # save config
    dest_dir, cat = os.path.split(prd_file)
    cat = cat.split('.')[0]
    dest_dir = os.path.join(dest_dir, clean_name)
    dest_alg_path = f"{dest_dir}/%s.align" % cat
    dest_cfg_file = f"{dest_dir}/%s.clean_config" % clean_name

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    save_config = [
        'prd_file', 'rvw_file', 'min_rvw_len', 'max_rvw_len', 'min_rvw_num',
        'max_rvw_num', 'min_rvw_positive', 'force_rvw_img'
    ]
    write_list_to_file(
        sum([[i + ':', eval('str(%s)' % i), ''] for i in save_config], []),
        dest_cfg_file,
        join=False
    )

    # clean and save file
    clean_generator = clean_data()
    write_list_to_json(clean_generator, dest_alg_path)
