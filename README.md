# Review Helpfulness Prediction

Our code is developed based on the open-source project [MatchZoo](https://github.com/NTMC-Community/MatchZoo-py).

## Requirements

We use python version 3.7 and the main dependent libs are listed in `requirements.txt`

```bash
conda create -f environment.yml
```

while some other requirements need to be installed handly

**ICU Tokenizer**

```bash
# conda install icu libarary
conda install icu pkg-config

# Or if you wish to use the latest version of the ICU library,
# the conda-forge channel typically contains a more up to date version.
conda install -c conda-forge icu

# mac os
CFLAGS="-std=c++11" PATH="/usr/local/opt/icu4c/bin:$PATH" \
    pip install ICU-Tokenizer

# ubuntu
CFLAGS="-std=c++11" pip install ICU-Tokenizer
```

**Emoji Translation**

```bash
git clone git@github.com:jhliu17/emoji.git
cd emoji
python setup.py install
```

## Data Processing

For processing text and image data, please read the details in [here](scripts/README.md).


## Run Code

Training, the experiment config settings are listed in `config` folder.

```bash
# Single gpu or dataparallel 
# [ckpt] is optional for continual training
sh scripts/train.sh device_ids config_file [ckpt]

# Or distributed training
sh scripts/train_dist.sh device_ids n_procs config_file [ckpt]
```

Evaluation

```bash
sh scripts/eval.sh device_ids config_file ckpt
```

## Ref Code

- Conv-KNRM [paper](http://www.cs.cmu.edu/~./callan/Papers/wsdm18-zhuyun-dai.pdf)
- SCAN [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)
- Fake-News [paper](https://arxiv.org/pdf/2010.03159v1.pdf)
