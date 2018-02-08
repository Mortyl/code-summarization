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
        self.UNK_TOKEN = '<UNK>'
        self.next_id = 0
        self.token_to_id = {}
        self.id_to_token = {}
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
                    #get rid of the <id> tags
                    code = list(filter(lambda t: t != "<id>" and t != "</id>", item.split()))
                    token_counter.update(code)

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
            temp.append([self.get_id_or_unk(token) for token in tokens.split()])

        assert len(source) == len(temp)

        return temp
            


class Corpus:

    def __init__(self, names, code):
    
        #self.SUBTOKEN_START = "<START>"
        #self.SUBTOKEN_END = "<END>"
        self.NONE = "<NONE>" #is this the same as padding?

    """def get_file_data(self, input_file):
        with open(input_file, 'r') as f:
            data = json.load(f)
        names = []
        original_names = []
        code = []
        for entry in data:
            # skip entries with no relevant data (this will crash the code)
            if len(entry["tokens"]) == 0 or len(entry["name"]) == 0:
                continue
            code.append(self.remove_id_markers(entry["tokens"]))
            original_names.append(",".join(entry["name"]))
            subtokens = entry["name"]
            names.append([self.SUBTOKEN_START] + subtokens + [self.SUBTOKEN_END])

        return names, code, original_names"""

    def remove_id_markers(self, code):
        return list(filter(lambda t: t != "<id>" and t != "</id>", code))