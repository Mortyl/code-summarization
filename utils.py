import torch
import torch.utils.data
from tqdm import tqdm
import itertools
from typing import Tuple
import pandas as pd

def get_data(path: str, limit: int) -> Tuple[list, list]:
    """
    Given a path to a .csv file with two columns, first containing the method names and
    the second containing the method bodies

    Limit is the amount of examples we want, a limit of 0 or less means we get all.
    """
    assert type(path) is str
    assert type(limit) is int

    data = pd.read_csv(path, header=None)

    names = data[0].values.tolist()
    bodies = data[1].values.tolist()

    if limit > 0:
        names = names[:limit]
        bodies = bodies[:limit]

    assert len(names) == len(bodies), f'Not equal amount of names and bodies! {len(names)} names, {len(bodies)} bodies.'

    #split into tokens
    for i, n in enumerate(tqdm(names, desc='Splitting names')):
        names[i] = n.split()

    for i, b in enumerate(tqdm(bodies, desc='Splitting bodies')):
        #get rid of the <id> tags
        bodies[i] = list(filter(lambda t: t != "<id>" and t != "</id>", b.split()))

    _bodies = []
    _prev = []
    _targets = []

    for i, name in enumerate(tqdm(names, desc='Creating examples')):
        for j, n in enumerate(name[1:], start=1):
            _bodies.append(bodies[i])
            _prev.append(name[:j])
            _targets.append(n)
            

    return _bodies, _prev, _targets

def filter_max_length(source: list, max_length: int) -> list:
    """
    Source is a list of strings
    Want to split the strings to get tokens and then any longer than max_length
    We truncate
    """
    assert type(source) is list
    assert type(max_length) is int

    temp = []
    for tokens in tqdm(source, desc='Filtering length'):
        temp.append(tokens[:max_length])
    return temp

def get_lengths(tensor: torch.LongTensor, pad_idx: int) -> torch.LongTensor:
    """
    Gets a 2d tensor, calculates how long the true sequence is.
    TODO: Make it use n-d tensors with a dim argument
    """
    assert type(tensor) is torch.LongTensor
    assert type(pad_idx) is int
    assert len(tensor.shape) == 2

    lengths = torch.LongTensor(tensor.shape[0])

    #for each row (i.e. example in the batch) get the length of elements that
    #aren't padding, we should NEVER get padding between legit tokens
    for i, row in enumerate(tensor):
        lengths[i] = len(list(itertools.filterfalse(lambda x: x == pad_idx, row)))

    return lengths

def get_mask(lengths: torch.LongTensor, max_length: int, mask_value: float) -> torch.FloatTensor:
    """
    Take in a tensor of lengths (ints) and create a mask that is [batch_size, max_length], where
    the first 'length' elements are 1 and the following max_length-length elements are mask_value
    """
    assert type(lengths) is torch.LongTensor
    assert type(max_length) is int
    assert type(mask_value) is float

    mask = torch.ones(lengths.shape[0], max_length)

    for i, length in enumerate(lengths):
        if length < max_length: #TODO: so below doesn't get empty slice error, maybe fix?
            mask[i][length:].fill_(mask_value)

    return mask

class SubtokenDataset(torch.utils.data.Dataset):

    def __init__(self, bodies, prev, target):
        """
        The standard PyTorch dataset objects only handle single-input-single-output data loading
        This handles multi-input-single-output
        """

        assert len(bodies) == len(prev)
        assert len(prev) == len(target)

        self.bodies = bodies
        self.prev = prev
        self.target = target


    def __len__(self):
        return len(self.bodies)

    def __getitem__(self, idx):
        #self.target[idx] needs to be wrapped in list!
        return torch.LongTensor(self.bodies[idx]), torch.LongTensor(self.prev[idx]), torch.LongTensor([self.target[idx]])


def f1_score(pred: torch.LongTensor, target: torch.LongTensor) -> Tuple[float, float, float]:
    """
    TODO: Make sure this works properly!
    TODO: Make this work without converting to lists! 

    Takes two LongTensors and produces the f1-score, precision and recall.
    """
    assert type(pred) is torch.LongTensor
    assert type(target) is torch.LongTensor

    pred = list(pred)
    target = list(target)

    tp = 0
    _target = target[:] #make a copy so we don't delete the actual list

    for p in pred:
        if p in _target:
            tp += 1
            target
            del _target[_target.index(p)]

    precision = tp/len(pred)

    _pred = pred[:] #make another copy

    tp = 0 #i don't think we need to calculate this again?

    for t in target:
        if t in _pred:
            tp += 1 #i don't think we need to calculate this again?
            del _pred[_pred.index(t)]

    recall = tp/len(target)

    f1 = 2 * (precision*recall)/(precision+recall)

    return f1, precision, recall

class PadCollate:

    def __init__(self, dim=0):
        self.dim=dim

    def pad_collate(self, batch):
        #find longest sequence
        bod_max_len = max(list(map(lambda x: x[0].shape[self.dim], batch)))
        prev_max_len = max(list(map(lambda x: x[1].shape[self.dim], batch)))
        _b = []
        _p = []

        for i,(bod, prev, tar) in enumerate(batch):
            _b.append(torch.cat((bod, torch.zeros([bod_max_len-bod.shape[self.dim]]).long()),dim=0))
            _p.append(torch.cat((prev, torch.zeros([prev_max_len-prev.shape[self.dim]]).long()),dim=0))
            assert len(_b[i]) == bod_max_len
            assert len(_p[i]) == prev_max_len
        
        #stack all
        x1s = torch.stack(list(map(lambda x: x, _b)), dim=0)
        x2s = torch.stack(list(map(lambda x: x, _p)), dim=0)
        ys = torch.stack(list(map(lambda x: x[2], batch)), dim=0)

        return x1s, x2s, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
    

