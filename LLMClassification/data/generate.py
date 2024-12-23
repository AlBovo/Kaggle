import pandas as pd

def load_csv(train = True):
    if train:
        return pd.read_csv("data/train.csv")
    return pd.read_csv("data/test.csv")
