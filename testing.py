import os
import sys


gpu_id = sys.argv[1]
split_id = gpu_id.split(',')

# test 1: modeling testing
os.system("sh scripts/train.sh %d tests/test_config.json" % int(split_id[0]))

# test 2: data parallel modeling testing
# os.system("sh scripts/train.sh %s tests/test_dp_config.json" % gpu_id)

# test 3: distributed data parallel modeling testing
# os.system("sh scripts/train_dist.sh %s %d tests/test_dp_config.json" % (gpu_id, len(split_id)))
