import json


def parse_aligned(input_file):
    with open(input_file, "r") as reader:
        for line in reader:
            data = json.loads(line)
            product = data['product']
            reviews = data['review']
            yield product, reviews


def parse_separated_aligned(product_file, review_file, product_ley, review_key):

    def flatten_list(samplelist, key):
        sample_return = []
        for sample in samplelist:
            sample_return.append(
                {k: v for k, v in zip(key, sample.split('!qp!'))})
        return sample_return

    with open(product_file, "r") as prd_reader, open(review_file, "r") as rvw_reader:
        for prd_line, rvw_line in zip(prd_reader, rvw_reader):
            product_info = json.loads(prd_line)
            reviews_info = json.loads(rvw_line)
            assert product_info['product_id'] == reviews_info['product_id']

            product_id = product_info['product_id']
            product = product_info['product']
            reviews = reviews_info['review']
            product_return = flatten_list(
                product,
                product_ley
            )
            reviews_return = flatten_list(
                reviews,
                review_key
            )

            yield product_id, product_return, reviews_return


def parse_file(input_files, expect_term_num=None, split_token=None, head_num=None):
    """Parse the lazada source file to a correct format line data.
    Since some special token may break the data into multiple lines. This function is
    built to solve this problem.

    Args:
        input_files (str): product table or review table file address.
        expect_term_num (int, optional): the expect term number. Defaults to None.
        split_token (str, optional): split token. Defaults to None.
        head_num (int, optional): for testing, early existing. Defaults to None.

    Yields:
        str: a correct format line.
    """
    if isinstance(input_files, str):
        file_list = [input_files]
    else:
        file_list = input_files

    for input_file in file_list:
        with open(input_file, "r") as reader:
            last_line = ''
            for line_idx, line in enumerate(reader):
                line = line.strip()  # strip the special character
                if expect_term_num is None:
                    yield line
                else:
                    # check if the line is broken by some special character (such as ^M, \n)
                    line = last_line + ' ' + line if last_line else line
                    d = line.split(split_token)
                    if len(d) == expect_term_num:
                        last_line = ''
                        yield line
                    else:
                        last_line = line
                
                if head_num and line_idx > head_num:
                    break


def parse_key(key_file, return_dict=True):
    """Parse lazada index key file.

    Args:
        key_file (str): key file path.
        return_dict (bool, optional): which format wants to return. Defaults to True.

    Returns:
        (dict, list): return key dict or key list.
    """
    prd = []
    rvw = []
    with open(key_file) as f:
        prd = f.readline().rstrip().split(',')
        rvw = f.readline().rstrip().split(',')

    if return_dict:
        prd_dict = {k.lstrip(' '): i for i, k in enumerate(prd)}
        rvw_dict = {k.lstrip(' '): i for i, k in enumerate(rvw)}
        return prd_dict, rvw_dict
    else:
        prd_list = [k.lstrip(' ') for k in prd]
        rvw_list = [k.lstrip(' ') for k in rvw]
        return prd_list, rvw_list


def parse_url(string) -> list:
    try:
        url = json.loads(string)
    except json.JSONDecodeError:
        return []
    
    if isinstance(url, str):
        url = [url]
    return url


def match_category(desire_cats: list, prd_cat: str):
    c = prd_cat.lower()
    for d in desire_cats:
        if d.lower() in c:
            return True
    return False


def sort_product_key(prd_key):
    sorted_key = sorted(prd_key)
    return sorted_key


def have_upvote(reviews, threshold):
    for review in reviews:
        if int(review['upvotes']) > threshold:
            return True
    return False
