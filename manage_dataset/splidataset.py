import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../dataset/daSplittare.tsv", sep='\t', encoding='ISO-8859-1')
train, test = train_test_split(df, test_size=0.2)

train.to_csv('../dataset/testSet.tsv', index=False, sep='\t')
test.to_csv('../dataset/trainingSet.tsv', index=False, sep='\t')

