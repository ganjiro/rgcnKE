import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(test_size=0.2, path=None):
    df = pd.read_csv(str(path) + "\\daSplittare.tsv", sep='\t', encoding='ISO-8859-1')
    train, test = train_test_split(df, test_size=test_size)

    train.to_csv(str(path) + '\\testSet.tsv', index=False, sep='\t')
    test.to_csv(str(path) + '\\trainingSet.tsv', index=False, sep='\t')
