import json
import datetime
from multiprocessing import Pool


def write_list_to_json(precessed_examples, dest_file):
    """
    Write a list containing type data to a json file.
    With json.dumps
    """
    def dumps(examples):
        for line in examples:
            yield json.dumps(line)

    line_num = 0
    with open(dest_file, 'w') as f:
        for e in dumps(precessed_examples):
            f.write(e + '\n')
            line_num += 1
    return line_num


def write_list_to_file(precessed_examples, dest_file, join=True, join_token="!qp!"):
    """
    Only write the origin data.

    Without json.dumps
    """
    def dumps(examples):
        for line in examples:
            if join:
                yield join_token.join(str(i) for i in line)
            else:
                yield line

    line_num = 0
    with open(dest_file, 'w') as f:
        for e in dumps(precessed_examples):
            f.write(e + '\n')
            line_num += 1
    return line_num


def filter_cat_review_length(review_string, min_length=None, max_length=None):
    """
    Filter cat review in extraction stage.
    """
    value = True
    if len(review_string) == 0:
        value = False
    
    if min_length and len(review_string) < min_length:
        value = False
    
    if max_length and len(review_string) > max_length:
        value = False

    return value


def filter_cat_date(date: str, min_date=None, max_date=None):
    value = True

    # process review date
    try:
        date = date.strip().split(' ')[0].split('-')  # get year, month, day
        date = datetime.date(*(int(i) for i in date))
    except:
        date = min_date
        value = False

    if min_date and date < min_date:
        value = False
    
    if max_date and date > max_date:
        value = False
    
    return value


def get_config_string(variable_list: list):
    assert len(variable_list) > 1
    return sum([[i + ':', eval('str(%s)' % i), ''] for i in variable_list], [])


class ProcessUnit:
    def __init__(self, max_worker: int = 1, chunksize: int = 1000000):
        self.worker = max_worker
        self.chunksize = chunksize
    
    def build_task(self, task_func, dataloader):
        if self.worker > 1:
            with Pool(self.worker) as pool:
                results = pool.imap_unordered(task_func, dataloader, chunksize=self.chunksize)
                for ret in results:
                    if ret:
                        yield ret
        else:
            for data in dataloader:
                ret = task_func(data)
                if ret:
                    yield ret
