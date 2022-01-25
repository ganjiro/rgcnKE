import pandas as pd

df = pd.read_csv("./sharable/TSV FINAL.txt", sep='\t')

df.unique()

df.to_csv('new_name.tsv',index=False)