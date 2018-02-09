#imports 
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.autograd import Variable

import vocab
import utils
import model

preprocess_start_time = time.time()

#constants
TEST_DATA = 'data/test.csv'
TRAIN_DATA = 'data/train.csv'
MAX_LENGTH = 300
THRESHOLD = 3
BATCH_SIZE = 32
USE_CUDA = torch.cuda.is_available()
N_EPOCHS = 10
MASK_VALUE = 1e-10

#import data w/ pandas
train_names, train_bodies = utils.get_data(TRAIN_DATA)
test_names, test_bodies = utils.get_data(TEST_DATA)

#filter out ones that are way too large by truncating the >MAX_LENGTH tokens
train_names = utils.filter_max_length(train_names, MAX_LENGTH)
train_bodies = utils.filter_max_length(train_bodies, MAX_LENGTH)

test_names = utils.filter_max_length(test_names, MAX_LENGTH)
test_bodies = utils.filter_max_length(test_bodies, MAX_LENGTH)

#create dictionary object 
dictionary = vocab.Dictionary()

#builds the token -> id mapping dict, only if tokens appear >= threshold times
dictionary.build_dict([train_names, train_bodies, test_names, test_bodies], THRESHOLD)

#tokenize sources
train_names_idx = dictionary.tokenize(train_names)
train_bodies_idx = dictionary.tokenize(train_bodies)

test_names_idx = dictionary.tokenize(test_names)
test_bodies_idx = dictionary.tokenize(test_bodies)

#pad sources
#if we DON'T shuffle the data in the dataloader then this is how we get lengths
train_names_idx, train_names_lengths = dictionary.pad_sequences(train_names_idx)
train_bodies_idx, train_bodies_lengths = dictionary.pad_sequences(train_bodies_idx)

test_names_idx, test_names_lengths = dictionary.pad_sequences(test_names_idx)
test_bodies_idx, test_bodies_lengths = dictionary.pad_sequences(test_bodies_idx)

#create dataset
train_names_idx = torch.LongTensor(train_names_idx)
train_bodies_idx = torch.LongTensor(train_bodies_idx)

test_names_idx = torch.LongTensor(test_names_idx)
test_bodies_idx = torch.LongTensor(test_bodies_idx)

#bodies are X, names are y
train_dataset = torch.utils.data.TensorDataset(train_bodies_idx, train_names_idx) 
test_dataset = torch.utils.data.TensorDataset(test_bodies_idx, test_names_idx)

#create dataloader
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

#define model, criterion and optimizer
model = model.ConvAttn()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)

if USE_CUDA:
    model = model.cuda()
    criterion = criterion.cuda()

#train loop
pad_idx = dictionary.get_pad_idx() #get this for calculating lengths per batch

print(f'Processing complete in {time.time() - preprocess_start_time:.1f}s')

assert 1==2

for e in range(1, N_EPOCHS+1):

    epoch_loss = 0
    epoch_start_time = time.time()
    start_time = time.time()

    #TODO: wrap this into nice 'train' function

    model.train()

    for i, (X, y) in tqdm(enumerate(train_data_loader, start=1), desc='Train'):

        #if we shuffle the data in the dataloader then this is how we get lengths
        #TODO: this currently adds on 14 seconds per epoch, ~2 mins for 10 epochs
        #TODO: need to weight up masking vs shuffling vs speed
        #lengths = utils.get_lengths(X, pad_idx)
        #mask = utils.get_mask(lengths, MAX_LENGTH, MASK_VALUE)

        X = Variable(X)
        y = Variable(y)

        if USE_CUDA:
            X = X.cuda()
            y = y.cuda()

        optimizer.zero_grad()

        _y = model(X)

        loss = criterion(_y, y)

        epoch_loss += loss.data[0]

        loss.backward()

        optimizer.step()

    #TODO: wrap this into nice eval/validate function

    model.eval()

    for i, (X, y) in tqdm(enumerate(train_data_loader, start=1), desc='Validate'):

        X = Variable(X, volatile=True)
        y = Variable(y, volatile=True)

        if USE_CUDA:
            X = X.cuda()
            y = y.cuda()

        _y = model(X)

        loss = criterion(_y, y)

        val_loss += loss.data[0]

    print(f'Epoch: {e}, Train Loss: {epoch_loss/len(train_data_loader)}, Val. Loss: {val_loss/len(val_data_loader)}, ms/epoch: {time.time() - epoch_start_time}')
    
    