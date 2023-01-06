import json
import os
import spacy
import neuralcoref
import re
import numpy as np
import pandas
import argparse
from spacy.lang.en import English
from spacy.pipeline import DependencyParser
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
from collections import defaultdict
#from utils import (
#    write_list_to_json,
#    write_list_to_file
#)

import sys
sys.path.append('/home/wh.347426/MCR')

from matchzoo import DataPack
LANG = 'en'

default_punc_chars = ['!', '.', '?', ':']

parser = argparse.ArgumentParser()
parser.add_argument('--cat', type=str, default='home_and_kitchen',  help="dataset category to be processed")
parser.add_argument('--setn', type=int, default=2, help="the set number of dataset")

args = parser.parse_args()

# modify this according to dataset setting
cat = args.cat
set_num = args.setn

data_dir = f"/home/wh.347426/dataset/amazon18/{cat}/set{set_num}/"
dill_dir = f"{data_dir}/all_rvw"
dst_dir = f"{data_dir}/all_rvw_new"

id_map_dir = f"{data_dir}/id_map"

data_prd_file = f"{data_dir}/{cat}.prd.data"
data_rvw_file = f"{data_dir}/{cat}.rvw.data"
prd_idmap_file = f"{id_map_dir}/prd_id_map.json"
rvw_idmap_file = f"{id_map_dir}/rvw_id_map.json"
dp_idmap_prefix = f"{id_map_dir}/dp_id_map"

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
    string = re.sub(r",", " ,", string)
    string = re.sub(r"\.\s", " . ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " !", string)
    string = re.sub(r"\(", " (", string)
    string = re.sub(r"\)", " )", string)
    string = re.sub(r"\?", " ?", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_dpid_dict(file_prefix, split = 'train'):
    '''generate pseudo-id--real-id mapping dictionary for right text'''
    file_name = file_prefix + '.' + split + '.json'
    if os.path.exists(file_name): 
        with open(file_name, 'r') as f:
            id_map = json.load(f)
    else:
        if not os.path.exists(id_map_dir):
            os.mkdir(id_map_dir)
        id_map = {}
        with open(file_name, 'w') as f:
            dp = DataPack.load(dill_dir, split)
            frame_slice = dp._right['id_right']
            for i, id_right in enumerate(tqdm(frame_slice)):
                id_map[id_right] = i
            json.dump(id_map, f)
    return id_map

def load_jsid_dict(data_file, json_file, key='product_id'):
    '''generate pseudo-id--real-id mapping dictionary for product id'''
    if os.path.exists(json_file): id_map = json.load(open(json_file, 'rb'))
    else:
        if not os.path.exists(id_map_dir):
            os.mkdir(id_map_dir)
        id_map = {}
        with open(json_file, 'w') as f:
            with open(data_file, 'r') as ff:
                js = json.load(ff)
                for idx, prd_id in enumerate(tqdm(js[key].values())):
                    id_map[prd_id] = str(idx)
                json.dump(id_map, f)
    return id_map

def get_senticizer():
    pass

def init_processor():
    # nlp_ = spacy.load("en")
    # nlp_ = spacy.load("en")
    # parser = nlp_.parser    

    nlp = English()

    # parser.vocab = nlp.vocb

    nlp.tokenizer = Tokenizer(nlp.vocab)
    nlp.add_pipe(nlp.create_pipe("sentencizer"), name="sentencizer")
    # nlp.add_pipe(parser, name="parser")    
    coref = neuralcoref.NeuralCoref(nlp.vocab)
    nlp.add_pipe(coref, name='neuralcoref')
    return nlp

def fix_sent(sent):
    sent = re.sub(r"i 'm", "i'm", sent)
    sent = re.sub(r"(i|we|they|you) 've", r"\1've", sent)
    sent = re.sub(r"(it|he|she) 's", r"\1's", sent) 
    return sent

def find_root(doc):
    for token in doc:
        if token.head == token:
            return token.lemma_.lower()

def main():
    with open(data_prd_file, 'r') as f: prd_js = json.load(f)
    # with open(data_rvw_file, 'r') as f: rvw_js = json.load(f)
    print("keys of prd_js {}:".format(prd_js.keys()))
    # print("keys of rvw_js {}:".format(rvw_js.keys()))
    
    # process each dill file
    # rvw_jsid_map = load_jsid_dict(data_rvw_file, rvw_idmap_file, 'review_id')
    prd_jsid_map = load_jsid_dict(data_prd_file, prd_idmap_file, 'product_id')

    word_idmap = DataPack.load(dill_dir, 'pipeline')
    rvw_vocab = word_idmap['rvw_text_field']['context_save']['vocab_unit']
    nlp = init_processor()
    nlpp = spacy.load('en')

    for split in ['train', 'dev', 'test']:
        print('begin processing %s set......'%split)
        dp = DataPack.load(dill_dir, split)
        if "attention_mask" not in dp.right.keys():
            dp._right.insert(len(dp._right.columns), "attention_mask", value=None)
        
        for idx, row in tqdm(dp.frame[:].iterrows()):
            prd_id = row['id_left']
            rvw_id = row['id_right']
	   
            #if prd_id=='B005GM15HE' and rvw_id=='B005GM15HE-260':
            #    import ipdb; ipdb.set_trace()
            #else:
            #    continue
            prd_name = prd_js['name'][prd_jsid_map[prd_id]]
            # rvw_text = rvw_js['content'][rvw_jsid_map[rvw_id]]
            rvw_text_idx = dp.right.text_right[rvw_id]
            rvw_text_tokens = rvw_vocab.detransform(rvw_text_idx)    

            rvw_text = ' '.join(rvw_text_tokens)
            doc = nlp(rvw_text)

            prd_name_processed = nlpp(prd_name)
            prd_name_short = find_root(prd_name_processed)
 
            try:
                assert len(doc) == len(rvw_text_idx)
            except:
                tks = [tk for tk in doc]
                print(tks)
                print(rvw_text_tokens)
                for i in range(len(tks)):
                    if str(tks[i]) != rvw_text_tokens[i]:
                        print("Not matching pos:")
                        print(tks[i:i+2], rvw_text_tokens[i:i+2])
                        break
                import ipdb; ipdb.set_trace()
            sent_starts = [0]
            sents = doc.sents
            for sent in sents:
                sent_starts.append(sent_starts[-1]+len(sent))
           
 
            # assign all sentences with coreference to be 1
            mask = np.array([False for _ in range(len(doc))], dtype=np.bool)
            if doc._.coref_clusters:
                corefs = doc._.coref_clusters[0]	    
                pts = 0
                for coref in corefs:
                    start = coref.start
                    while coref.start >= sent_starts[pts + 1]: pts = pts + 1
                    mask[sent_starts[pts] : sent_starts[pts + 1]] = True
            
            sents = doc.sents
            for i,sent in enumerate(sents):
                if prd_name_short in str(sent):
                    mask[sent_starts[i] : sent_starts[i + 1]] = True

            dp._right.attention_mask[rvw_id] = mask 

        # dp._right['attention_mask'] = mask_list
        dp.save(dirpath=dst_dir, name=f"{split}")
        print('end processing %s set......'%split)

if __name__ == "__main__":
    main()
