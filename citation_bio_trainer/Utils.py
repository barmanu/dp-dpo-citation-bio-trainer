import json


def load_json(path):
    d = None
    with open(path) as f:
        d = json.load(f)
    f.close()
    return d


def change_nl(x):
    if "\n" in x:
        return "MWLN"
    else:
        return x
