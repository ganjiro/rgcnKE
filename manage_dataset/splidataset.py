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

    train.to_csv(str(path) + '\\testSet.tsv', index=False, sep='\t')
    test.to_csv(str(path) + '\\trainingSet.tsv', index=False, sep='\t')
    fix_quotation(str(path) + '\\testSet.tsv', True)
    fix_quotation(str(path) + '\\trainingSet.tsv', True)

    # np.save(str(path) + '\\train_idx.npy', train)
    # np.save(str(path) + '\\test_idx.npy', test)
