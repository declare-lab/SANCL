import json
import gzip

from tqdm import tqdm


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def parse_json(input_file):
    with open(input_file, "r") as reader:
        for l in reader:
            yield json.loads(l)


def process_review(data_file):
    processed_data = []

    print('Process review...')
    for i, l in enumerate(tqdm(parse(data_file))):
        review = l['reviewText']
        # no review
        if len(review) == 0:
            continue

        helpful = l['helpful']
        # no voting
        if helpful[1] == 0:
            continue

        data = {}
        data['asin'] = l['asin']
        data['review_text'] = review
        score = helpful[0] / helpful[1]
        data['classification_label'] = 'helpful' if score >= 0.75 else 'unhelpful'
        data['regression_label'] = score
        data['upvotes'] = helpful[0]
        data['total_votes'] = helpful[1]
        data['review_text_len'] = len(review)

        processed_data.append(data)
    return processed_data


def process_metadata(data_file):
    processed_data = {}

    print('Process metadata...')
    for i, l in enumerate(tqdm(parse(data_file))):
        asin = l['asin']
        title = l.get('title', None)

        if title is not None and len(title) > 0:
            data = {}
            data['asin'] = asin
            data['product_text'] = title
            processed_data[asin] = data

    return processed_data


def align_metadata(product_data: dict, review_data: list):
    '''
    Align review to the product item
    '''

    print('Align metadata...')
    for r in tqdm(review_data):
        meta = product_data.get(r['asin'], None)

        if meta is not None:
            if 'review' in meta.keys():
                meta['review'].append(r)
            else:
                meta['review'] = [r]

    print('Delete empty product...')
    delete_key = []
    for k in product_data:
        if not 'review' in product_data[k].keys():
            delete_key.append(k)
    
    for k in delete_key:
        product_data.pop(k)
    
    return product_data


def write_list_to_json(precessed_examples, dest_file):
    def dumps(examples):
        for l in examples:
            yield json.dumps(l)

    line_num = 0
    with open(dest_file, 'w') as f:
        for e in dumps(precessed_examples):
            f.write(e + '\n')
            line_num+=1
    return line_num


def write_dict_to_json(precessed_examples, dest_file):
    def dumps(examples):
        for k, l in examples.items():
            yield json.dumps(l)

    with open(dest_file, 'w') as f:
        for e in dumps(precessed_examples):
            f.write(e + '\n')


if __name__ == '__main__':
    review_file = "/home/junhao.jh/dataset/amazon/reviews_Clothing_Shoes_and_Jewelry.json.gz"
    meta_file = "/home/junhao.jh/dataset/amazon/meta_Clothing_Shoes_and_Jewelry.json.gz"
    dest_file = "/home/junhao.jh/dataset/amazon/Clothing_Shoes_and_Jewelry.data"

    print(review_file)
    print(meta_file)
    print(dest_file)

    # process review and meta data
    review_data = process_review(review_file)
    meta_data = process_metadata(meta_file)
    aligned_data = align_metadata(meta_data, review_data)

    print('Product num: %d' % len(aligned_data))
    write_dict_to_json(aligned_data, dest_file)
    
    # for i in parse(data_file):
    #     print(i)
