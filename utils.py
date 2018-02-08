from tqdm import tqdm

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
        temp.append(' '.join(tokens.split()[:max_length]))
    return temp
