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
import models

preprocess_start_time = time.time()

#constants
LIMIT = 10

TEST_DATA = 'data/test.csv'
TRAIN_DATA = 'data/train.csv'
MAX_LENGTH = 300
THRESHOLD = 3
BATCH_SIZE = 32
USE_CUDA = torch.cuda.is_available()

EMBEDDING_DIM = 128
#for conv
K1 = 8
K2 = 8
W1 = 24
W2 = 29
W3 = 10

N_EPOCHS = 10
MASK_VALUE = 1e-10

#import data w/ pandas
train_bodies, train_names_prev, train_names_targets = utils.get_data(TRAIN_DATA, LIMIT)
test_bodies, test_names_prev, test_names_targets  = utils.get_data(TEST_DATA, LIMIT)

#truncating bodies with >MAX_LENGTH tokens
train_bodies = utils.filter_max_length(train_bodies, MAX_LENGTH)
test_bodies = utils.filter_max_length(test_bodies, MAX_LENGTH)

#create dictionary object 
dictionary = vocab.Dictionary()

#builds the token -> id mapping dict, only if tokens appear >= threshold times
dictionary.build_dict([train_bodies, train_names_prev, train_names_targets, test_bodies, test_names_prev, test_names_targets], THRESHOLD)

vocab_size = len(dictionary.token_to_id)

#tokenize sources
train_bodies_idx = dictionary.tokenize(train_bodies)
train_names_prev_idx = dictionary.tokenize(train_names_prev)
train_names_targets_idx = dictionary.tokenize(train_names_targets)

test_bodies_idx = dictionary.tokenize(test_bodies)
test_names_prev_idx = dictionary.tokenize(test_names_prev)
test_names_targets_idx = dictionary.tokenize(test_names_targets)

#pad sources
#if we DON'T shuffle the data in the dataloader then this is how we get lengths
train_bodies_idx, _ = dictionary.pad_sequences(train_bodies_idx)
train_names_prev_idx, _ = dictionary.pad_sequences(train_names_prev_idx)

test_bodies_idx, _ = dictionary.pad_sequences(test_bodies_idx)
test_names_prev_idx, _ = dictionary.pad_sequences(test_names_prev_idx)

#create dataset
#train_bodies_idx = torch.LongTensor(train_bodies_idx)
#train_names_prev_idx = torch.LongTensor(train_names_prev_idx)
#train_names_target_idx = torch.LongTensor(train_names_target_idx)

#test_bodies_idx = torch.LongTensor(test_bodies_idx)
#test_names_prev_idx = torch.LongTensor(test_names_prev_idx)
#test_names_target_idx = torch.LongTensor(test_names_target_idx)

#bodies are X, names are y
train_dataset = utils.SubtokenDataset(train_bodies_idx, train_names_prev_idx, train_names_targets_idx) 
test_dataset = utils.SubtokenDataset(test_bodies_idx, test_names_prev_idx, test_names_targets_idx)

#create dataloader
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

#define model, criterion and optimizer
model = models.ConvAttentionNetwork(vocab_size, EMBEDDING_DIM, K1, K2, W1, W2, W3)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters())
#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)

if USE_CUDA:
    model = model.cuda()
    criterion = criterion.cuda()

#train loop
pad_idx = dictionary.get_pad_idx() #get this for calculating lengths per batch

assert pad_idx == 0, 'PADDING CURRENTLY NEEDS TO BE ZERO FOR THE MODEL TO WORK'

print(f'Processing complete in {time.time() - preprocess_start_time:.1f}s')


for e in range(1, N_EPOCHS+1):

    epoch_loss = 0
    val_loss = 0
    epoch_start_time = time.time()
    start_time = time.time()

    #TODO: wrap this into nice 'train' function

    model.train()

    for i, (Xb, Xn, y) in enumerate(tqdm(train_data_loader, desc='   Train'), start=1):

        #if we shuffle the data in the dataloader then this is how we get lengths
        #TODO: this currently adds on 14 seconds per epoch, ~2 mins for 10 epochs
        #TODO: need to weight up masking vs shuffling vs speed
        #lengths = utils.get_lengths(X, pad_idx)
        #mask = utils.get_mask(lengths, MAX_LENGTH, MASK_VALUE)

        Xb = Variable(Xb)
        Xn = Variable(Xn)
        y = Variable(y.squeeze(1))

        if USE_CUDA:
            Xb = Xb.cuda()
            Xn = Xn.cuda()
            y = y.cuda()

        optimizer.zero_grad()

        _y = model.forward(Xb, Xn)

        loss = criterion(_y, y)

        epoch_loss += loss.data[0]

        loss.backward()

        optimizer.step()

    #TODO: wrap this into nice eval/validate function

    model.eval()

    """for i, (Xb, Xn, y) in enumerate(tqdm(train_data_loader, desc='Validate'), start=1):

        Xb = Variable(Xb, volatile=True)
        Xn = Variable(Xn, volatile=True)
        y = Variable(y.squeeze(1), volatile=True)

        if USE_CUDA:
            Xb = Xb.cuda()
            Xn = Xn.cuda()
            y = y.cuda()

        _y = model(Xb, Xn)

        loss = criterion(_y, y)

        val_loss += loss.data[0]"""

    print(f'   Epoch: {e}, Train Loss: {epoch_loss/len(train_data_loader)}, Val. Loss: {val_loss/len(train_data_loader)}, s/epoch: {time.time() - epoch_start_time}')
    
    