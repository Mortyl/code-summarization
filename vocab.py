import json
from collections import Counter
from itertools import chain
from tqdm import tqdm

class Dictionary:
    """
    A simple feature dictionary that can convert features (ids) to
    their textual representation and vice-versa.
    """

    def __init__(self):
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.next_id = 0
        self.token_to_id = {}
        self.id_to_token = {}
        self.add_or_get_id(self.get_pad())
        self.add_or_get_id(self.get_unk())

    def add_or_get_id(self, token):
        """
        input a token, if it exists in the dictionary already then this returns the id
        if it doesn't exist in the dictionary already then it adds it and returns the id
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        this_id = self.next_id
        self.next_id += 1
        self.token_to_id[token] = this_id
        self.id_to_token[this_id] = token

        return this_id

    def is_unk(self, token):
        """
        returns True if the token is <UNK>, else False
        """
        return token not in self.token_to_id

    def get_id_or_unk(self, token):
        """
        input a token, get the id if it exists in the dictionary, get the <UNK> token if it doesn't
        """
        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            return self.token_to_id[self.get_unk()]

    def get_id_or_none(self, token):
        """
        input a token, get the id if it exists in the dictionary, get None if it doesn't
        """
        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            return None

    def get_name_for_id(self, token_id):
        """
        go from id back to token
        """
        return self.id_to_token[token_id]

    def __len__(self):
        return len(self.token_to_id)

    def __str__(self):
        return str(self.token_to_id)

    def get_all_names(self):
        return frozenset(self.token_to_id.keys())

    def get_unk(self):
        return self.UNK_TOKEN

    def get_unk_idx(self):
        unk_idx = self.get_id_or_none(self.get_unk())
        assert unk_idx is not None
        return pad_idx

    def get_pad(self):
        return self.PAD_TOKEN

    def get_pad_idx(self):
        pad_idx = self.get_id_or_none(self.get_pad())
        assert pad_idx is not None
        return pad_idx

    def build_dict(self, sources: list, threshold: int=10) -> None:
        """
        Names and bodies share a dictionary, so this function allows you to put both in at once
        Sources is a list of 'source'
        Each 'source' is a list of strings
        The strings are space separated tokens
        """
        assert type(sources) is list
        assert type(threshold) is int

        token_counter = Counter()

        #build counter
        for source in sources: 
            for item in tqdm(source, desc='Building counter'):
                    token_counter.update(item)

        for token, count in tqdm(token_counter.items(), desc='Building dictionary'):
            if count >= threshold:
                self.add_or_get_id(token)

    def tokenize(self, source: list):
        """
        Source is a list of lists
        Those lists are the split tokens
        """
        assert type(source) is list

        temp = []

        for tokens in tqdm(source, desc='Tokenizing'):
            if type(source[0]) is list: #this is to let us tokenize things that aren't list of lists, i.e. targets will be list of ints
                temp.append([self.get_id_or_unk(token) for token in tokens])
            else:
                temp.append(self.get_id_or_unk(tokens))

        assert len(source) == len(temp)

        return temp

    def pad_sequences(self, source: list, max_length: int = 0) -> list:
        """
        Gets a list of indices and pads them to max length
        Sequence should already be trimmed!
        """
        assert type(source) is list
        assert type(source[0]) is list
        assert type(source[0][0]) is int, 'Sequences must already be indexed!'
        assert type(max_length) is int

        pad_idx = self.get_pad_idx()
        assert pad_idx is not None

        #if max_length is 0 or less, it means we calculate it
        if max_length <= 0:
            max_length = max(list(map(len, source)))

        #get the true value of the lengths before padding, to potentially use later
        lengths = list(map(len, source))

        temp = []
        

        for indices in tqdm(source, desc='Padding sequences'):
            assert len(indices) <= max_length, 'Indices should already be truncated to max_length!'
            indices = indices + [pad_idx for _ in  range(max_length - len(indices))]
            assert len(indices) == max_length, 'Padding error! Padded sequence != max_length!'
            temp.append(indices)

        assert len(lengths) == len(temp)
        assert len(source) == len(temp)

        return temp, lengths