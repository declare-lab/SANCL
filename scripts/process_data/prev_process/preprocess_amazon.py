#!/usr/bin/env python
# coding:utf-8
import os
import json
import re
import numpy as np

from tqdm import tqdm
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from sklearn.model_selection import train_test_split

"""
WoS Reference: https://github.com/kk7nc/HDLTex
"""

FILE_DIR = '/home/junhao.jh/dataset/amazon/Clothing_Shoes_and_Jewelry.data'
PARA_DIR, FILE = os.path.split(FILE_DIR)
total_len = []
np.random.seed(7)

# nlp = English()
# Create a blank Tokenizer with just the English vocab
# tokenizer = Tokenizer(nlp.vocab)
nlp = spacy.load('en_core_web_lg')
import ipdb; ipdb.set_trace()

english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]


def write_to_json(precessed_examples, dest_file):
    def dumps(examples):
        for line in examples:
            yield json.dumps(line)

    with open(dest_file, 'w') as f:
        for e in dumps(precessed_examples):
            f.write(e + '\n')


def write_to_file(processed_examples, dest_file):
    with open(dest_file, 'w') as f:
        f.writelines(processed_examples)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def tokenize_sentence(sent):
    sent = tokenizer(sent)
    return list([w.text for w in sent])


def post_processing(stage='train'):
    print('Flaten and process %s...' % stage)
    with open(os.path.join(PARA_DIR, '%s.%s' % (FILE, stage))) as f:
        data = []
        for line in tqdm(f):
            d = json.loads(line)
            d['label'] = [d['label']]
            data.append(d)
            
    write_to_json(data, os.path.join(PARA_DIR, '%s.%s' % (FILE, stage)))


def get_data_from_meta(stage='train'):
    print('Flaten and process %s...' % stage)
    with open(os.path.join(PARA_DIR, '%s.emb.%s' % (FILE, stage))) as f:
        data = []
        label_check = {}

        for line in tqdm(f):
            product = json.loads(line)
            product_text = product['product_text']
            product_text = tokenize_sentence(clean_str(product_text))

            for review in product['review']:
                review_text = review['review_text']
                label_text = review['classification_label']
                label_regr = review['regression_label']

                if label_text in label_check.keys():
                    label_check[label_text] += 1
                else:
                    label_check[label_text] = 1

                review_text = tokenize_sentence(clean_str(review_text))
                data.append({'product_token': product_text,
                             'review_token': review_text,
                             'classification_label': [label_text.lower()],
                             'regression_label': [label_regr],
                             'upvotes': [review['upvotes']],
                             'total_votes': [review['total_votes']]})

    print("label stat:", label_check)
    print("label nums:", len(label_check.keys()))

    write_to_json(data, os.path.join(PARA_DIR, '%s.%s' % (FILE, stage)))


def split_train_dev_test():
    f = open(FILE_DIR, 'r')
    data = f.readlines()
    f.close()
    # id = [i for i in range(len(data))]
    # np_data = np.array(data)
    # np.random.shuffle(id)
    # np_data = np_data[id]

    train, test = train_test_split(data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)

    train = list(train)
    val = list(val)
    test = list(test)
    write_to_file(train, os.path.join(PARA_DIR, '%s.emb.train' % FILE))
    write_to_file(val, os.path.join(PARA_DIR, '%s.emb.val' % FILE))
    write_to_file(test, os.path.join(PARA_DIR, '%s.emb.test' % FILE))

    print(len(train), len(val), len(test))
    return


if __name__ == '__main__':
    # split by product item
    print('Split data...')
    split_train_dev_test()

    # process data
    get_data_from_meta('train')
    get_data_from_meta('val')
    get_data_from_meta('test')
