from matchzoo.dataloader import InstanceDataset
from matchzoo import DataPack


# amazon clothing
save_dir = "/home/wh.347426/dataset/amazon18/clothing_shoes_and_jewelry/set4/all_rvw_bert"

# amazon home
# save_dir = "/home/wh.347426/dataset/amazon18/home_and_kitchen/set2/all_rvw"

# amazon electronics
# save_dir = "/home/wh.347426/dataset/amazon18/electronics/set2/all_rvw"

dup = 2
neg = 7
interval = 1
max_pos_samples = None

train_pack_processed = DataPack.load(save_dir, 'train')
InstanceDataset.generate_dataset(
    train_pack_processed,
    mode='pair',
    num_dup=dup,
    num_neg=neg,
    max_pos_samples=max_pos_samples,
    save_dir=save_dir,
    building_interval=interval,
    name='train'
)
