import torch
from tqdm import tqdm
import itertools
from typing import Tuple
import pandas as pd

def get_data(path: str) -> Tuple[list, list]:
    """
    Given a path to a .csv file with two columns, first containing the method names and
    the second containing the method bodies
    """
    assert type(path) is str

    data = pd.read_csv(path, header=None)

    names = data[0].values.tolist()
    bodies = data[1].values.tolist()

    assert len(names) == len(bodies), f'Not equal amount of names and bodies! {len(names)} names, {len(bodies)} bodies.'

    #split into tokens
    for i, n in enumerate(tqdm(names, desc='Splitting names')):
        names[i] = n.split()

    for i, b in enumerate(tqdm(bodies, desc='Splitting bodies')):
        bodies[i] = b.split()

    return names, bodies

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