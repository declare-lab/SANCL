# Dataset Processing

We crawl the multimodal review helpfulness dataset from Lazada.com and Amazon.com. The data processing procedures are summarized in the following.

## Processing Step

Step 1: Extract the paired product and review data from the raw log file

```
python scripts/process_data/extract_lazada.py
```

Step 2: Clean the product and review content

```
python scripts/process_data/clean_lazada.py
```

Step 3: Construct the dataset to pd.DataFrame style 

```
python scripts/process_data/construct_lazada.py
```

Step 4: Split the data into `train`, `dev`, and `test` fractions

```
python scripts/process_data/split_lazada.py
```

Step 5: Download the image data. Use the above fractions as the indicating files.

```
python scripts/download_data/collect_img.py
```

Step 6: Extract the image features via `bottom-up-attention`.

plz follow the installation and extraction code [bottom-up](https://github.com/MILVLG/bottom-up-attention.pytorch)


## Cache Data Reading

The `train.data.dill` file is the processed file including the text content of product and review. The `pipeline.dill` file is the processed vocabulary cache. To read the data, you can use the command belows.

```python
from matchzoo import DataPack

frame = DataPack.load('your_cache_processed_data_folder/', 'train')

# List the product details
frame.left

# List the review details
frame.right
```

## Coreference Mask Generation
To generate the selective mask, first install the required environment:
```
conda env create -f environment.yml
```
The code in the folder gen_mask works on '.dill' datapack file. You should first run the model to conduct once the preprocessing procedure to generate these datapack file from raw dataset file (.data/train/val/test), then run corresponding generate_mask file:
```
python gen_mask/gen_amazon.py --cat <category> --set <set_no>
```
