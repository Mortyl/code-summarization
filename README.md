# code-summarization

### DONE

- Download data from: http://groups.inf.ed.ac.uk/cup/codeattention/dataset.zip
- Place in /data folder and extract
    - Folder structure should be /data/json (also have train and test inside /data but we don't care about them)
- Run `preprocess.py` to get data into csv formats
    - Columns are: json filename, java filename, method name, method body
	- Method names are split on camelCase and snake_case
	- Method bodies are tokenized, with ids preceeded by an <id> token and followed by a </id> token

### TODO

- Re-create Conv Attention model in PyTorch(?)
- Train simple LSTM
- Train simple BiDAF
- Does the filename give any information? Are we allowed to use it? Do they use it?
- Use pygment to get type information, is this useful?
- Can we pre-train on another corpus? If we use type information, does the other corpus have to be Java?
	