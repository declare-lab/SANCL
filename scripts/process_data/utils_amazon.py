import json
import gzip


def parse_file(input_files, head_num: int = None):
    """Parse the amazon source file to a correct format line data.
    """
    if isinstance(input_files, str):
        file_list = [input_files]
    else:
        file_list = input_files

    for input_file in file_list:
        with open(input_file, "r") as reader:
            for line_idx, line in enumerate(reader):
                line = line.strip()  # strip the special character
                yield json.loads(line)
                if head_num and line_idx > head_num:
                    break


def parse_gzip_file(input_files, head_num: int = None):
    """Parse the amazon source file to a correct format line data.
    """
    if isinstance(input_files, str):
        file_list = [input_files]
    else:
        file_list = input_files

    for input_file in file_list:
        with gzip.open(input_file, 'rb') as reader:
            for line_idx, line in enumerate(reader):
                line = line.strip()  # strip the special character
                yield json.loads(line)
                if head_num and line_idx > head_num:
                    break


def parse_separated_aligned(product_file, review_file):
    with open(product_file, "r") as prd_reader, open(review_file, "r") as rvw_reader:
        for prd_line, rvw_line in zip(prd_reader, rvw_reader):
            product_info = json.loads(prd_line)
            reviews_info = json.loads(rvw_line)
            assert product_info['product_id'] == reviews_info['product_id']

            product_id = product_info['product_id']
            product = [json.loads(i) for i in product_info['product']]
            reviews = [json.loads(i) for i in reviews_info['review']]
            yield product_id, product, reviews


def have_upvote(reviews, threshold):
    for review in reviews:
        if int(review.get('vote', '0').replace(',', '').replace(' ', '')) > threshold:
            return True
    return False


def parse_url(string) -> list:
    try:
        url = json.loads(string)
    except json.JSONDecodeError:
        return []
    
    if isinstance(url, str):
        url = [url]
    return url
