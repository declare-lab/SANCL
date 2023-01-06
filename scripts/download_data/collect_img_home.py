from download import (
    ThreadPoolDownloader
)
from download_utils import (
    get_from_logger,
    get_rvw_prd_url,
    pic_statics
)


if __name__ == '__main__':
    splits = ['rvw', 'prd']  # ('rvw', 'prd')
    stages = ['train', 'dev', 'test']  # ('train', 'dev', 'test')
    max_workers = 100
    cat = 'home_prd_y18_19'
    dest_dir = f"/home/wh.347426/dataset/lazada/{cat}/set2"
    download_dir = f'{dest_dir}/pictures/%s/%s/'

    for stage in stages:
        for split in splits:
            prd_path = f'{dest_dir}/{cat}.prd.%s' % stage
            rvw_path = f'{dest_dir}/{cat}.rvw.%s' % stage
            logger = 'logger.txt'

            # download product pic
            # dir_path = download_dir % (stage, prd_dir)
            # prd_dl = ThreadPoolDownloader(
            #     max_workers,
            #     dir_path,
            #     logger
            # )
            # prd_to_do = get_rvw_prd_url(prd_path, 'product')
            # prd_dl.download(prd_to_do)

            # download review pic
            # dir_path = download_dir % (stage, rvw_dir)
            # rvw_dl = ThreadPoolDownloader(
            #     max_workers,
            #     dir_path,
            #     logger
            # )
            # rvw_to_do = get_rvw_prd_url(rvw_path, 'review')
            # total_pic_num = len(rvw_to_do)
            # rvw_dl.download(rvw_to_do)
            # rvw_to_do = get_from_logger(os.path.join(dir_path, logger))
            # rvw_dl.download(rvw_to_do)
            # down_pic_num = pic_statics(dir_path)
            # print(total_pic_num, down_pic_num)

            dir_path = download_dir % (stage, split)
            split_dl = ThreadPoolDownloader(
                max_workers,
                dir_path,
                logger
            )

            # from dataframe
            if split == 'prd':
                split_to_do = get_rvw_prd_url(prd_path, 'product', True)
            else:
                split_to_do = get_rvw_prd_url(rvw_path, 'review')
            total_pic_num = len(split_to_do)
            split_dl.download(split_to_do)

            # from logger
            split_to_do = get_from_logger(split_dl.log_file)
            split_dl.download(split_to_do)

            # stat pics number
            down_pic_num = pic_statics(dir_path)
            print(total_pic_num, down_pic_num)
