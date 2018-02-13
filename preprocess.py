import os
from tqdm import tqdm
import json
import pandas as pd

DATA_DIR = 'data/json/'

START_TOKEN = '<TITLE_START>'
END_TOKEN = '<TITLE_START/>'

ALL_FILES = os.listdir(DATA_DIR)

TRAIN_FILES = ['cassandra_train_methodnaming.json']
TEST_FILES = ['cassandra_test_methodnaming.json']

"""TRAIN_FILES = [file for file in ALL_FILES if 'train' in file and 'all' not in file and 'shuffled' not in file]

TEST_FILES = [file for file in ALL_FILES if 'test' in file and 'shuffled' not in file]"""

assert len(TRAIN_FILES) == len(TEST_FILES)

def files_to_examples(DIR, FILES):
    """
    DIR is the base directory containing all files
    FILES is the list of json files you want to get the metadata from
    """
    data = []
    for file in tqdm(FILES, desc='Loading data'):
        with open(f'{DIR+file}', 'r') as r:
            project = json.load(r)
            for method in project:
                assert type(file) is str
                assert type(method['filename']) is str
                assert type(method['name']) is list
                assert type(method['tokens']) == list
                method_name = START_TOKEN + ' ' + ' '.join(method['name']) + ' ' + END_TOKEN
                method_body = ' '.join(method['tokens'])
                data.append((method_name, method_body))
    return data

train_data = files_to_examples(DATA_DIR, TRAIN_FILES)
test_data = files_to_examples(DATA_DIR, TEST_FILES)

print(f'Training examples: {len(train_data)}')
print(f'Testing examples: {len(test_data)}')

train_df = pd.DataFrame.from_records(train_data)
test_df = pd.DataFrame.from_records(test_data)

train_df.to_csv('data/cassandra_train.csv', index=False, header=False)
test_df.to_csv('data/cassandra_test.csv', index=False, header=False)