import os
from random import randint

import pandas as pd
from sklearn.model_selection import train_test_split
from manage_dataset.reformat_data import fix_quotation
import numpy as np


def split_dataset(test_size=0.2, path=None):
    seed = randint(1, 1000)
    df = pd.read_csv(str(path) + "\\daSplittare.tsv", sep='\t', encoding='ISO-8859-1')
    train, test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    train.to_csv(str(path) + '\\test.txt', index=False, sep='\t')
    test.to_csv(str(path) + '\\train.txt', index=False, sep='\t')
    fix_quotation(str(path) + '\\test.txt', True)
    fix_quotation(str(path) + '\\train.txt', True)

    edges = str(path) + "\\edges.npz"

    if os.path.isfile(edges):
        os.remove(edges)