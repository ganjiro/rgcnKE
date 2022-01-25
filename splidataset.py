import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("km4city_test/daSplittare.tsv", sep='\t', encoding='ISO-8859-1')
train, test = train_test_split(df, test_size=0.2)

train.to_csv('km4city_test/testSet.tsv', index=False, sep='\t')
test.to_csv('km4city_test/trainingSet.tsv', index=False, sep='\t')

