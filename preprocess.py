import os
from tqdm import tqdm
import json
import pandas as pd

DATA_DIR = 'data/json/'

ALL_FILES = os.listdir(DATA_DIR)

TRAIN_FILES = [file for file in ALL_FILES if 'train' in file and 'all' not in file and 'shuffled' not in file]

TEST_FILES = [file for file in ALL_FILES if 'test' in file and 'shuffled' not in file] 

assert len(TRAIN_FILES) == len(TEST_FILES)
assert len(TRAIN_FILES) == 11

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
                data.append((file, method['filename'], ' '.join(method['name']), ' '.join(method['tokens'])))
    return data

train_data = files_to_examples(DATA_DIR, TRAIN_FILES)
test_data = files_to_examples(DATA_DIR, TEST_FILES)

print(f'Training examples: {len(train_data)}')
print(f'Testing examples: {len(test_data)}')

train_df = pd.DataFrame.from_records(train_data)
test_df = pd.DataFrame.from_records(test_data)

train_df.to_csv('data/train.csv', index=False, header=False)
test_df.to_csv('data/test.csv', index=False, header=False)