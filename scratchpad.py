#imports 
import pandas as pd

import torch

import vocab
import utils

#constants
TEST_DATA = 'data/test.csv'
TRAIN_DATA = 'data/train.csv'
MAX_LENGTH = 300
THRESHOLD = 3

#import data w/ pandas
data = pd.read_csv('data/train.csv', header=None)

names = data[0].values.tolist()
bodies = data[1].values.tolist()

assert len(names) == len(bodies)

#filter out ones that are way too large
bodies = utils.filter_max_length(bodies, MAX_LENGTH)

#create dictionary object 
dictionary = vocab.Dictionary()

#builds the token -> id mapping dict, only if tokens appear >= threshold times
dictionary.build_dict([names, bodies], THRESHOLD)

#tokenize sources
names_idx = dictionary.tokenize(names)
bodies_idx = dictionary.tokenize(bodies)

#TODO: create dataset (why is this not working)
train_dataset = torch.utils.data.TensorDataset(bodies_idx, names_idx)

#TODO: create dataloader

