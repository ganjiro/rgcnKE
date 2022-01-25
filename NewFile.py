import pandas as pd
import numpy as np

df = pd.read_csv("testfile.txt", sep='\t', encoding='ISO-8859-1')

df = df.drop_duplicates()
df1 = df.query('p != "<http://www.disit.org/km4city/schema#eventCategory>"')
df2 = df.query('p == "<http://www.disit.org/km4city/schema#eventCategory>"')


df.to_csv('km4city_test/total.tsv',index=False)
#df1.to_csv('km4c_stripped.nt',index=False, sep=' ')

df2.reset_index(drop=True, inplace=True)
df2.drop('p', inplace=True, axis=1)
df2.rename(columns={'s': 'nodes',
                   'o': 'label'},
          inplace=True, errors='raise')
df2.index = np.arange(1, len(df2)+1)
df2.index.name = 'id'
df2.to_csv('km4city_test/completeDataset.tsv', index=True, sep='\t')

df = pd.read_csv("km4city_test/completeDataset.tsv", sep='\t', encoding='ISO-8859-1')
df3 = df[['nodes', 'id', 'label']]
df3.index = np.arange(1, len(df3)+1)
df3.to_csv('km4city_test/daSplittare.tsv', index=False, sep='\t')

# file_name = 'km4city_test/km4c_stripped.nt'
# string_to_add = " ."
# with open(file_name, 'r',encoding='ISO-8859-1') as f:
#     file_lines = [''.join([x.strip(), string_to_add, '\n']) for x in f.readlines()]
#
# with open(file_name, 'w', encoding='ISO-8859-1') as f:
#     f.writelines(file_lines)
