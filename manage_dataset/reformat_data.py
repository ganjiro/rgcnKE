import pandas as pd
import numpy as np
import csv
import gzip

def fix_quotation(file_path, remove_brakets=False):
    with open(file_path, 'r', encoding="ISO-8859-1") as file:
        filedata = file.read()

    filedata = filedata.replace('#%#', '"')
    if remove_brakets:
        filedata = filedata.replace('<', '')
        filedata = filedata.replace('>', '')

    with open(file_path, 'w', encoding="ISO-8859-1") as file:
        file.write(filedata)

def reformat_data():
    with open("../dataset/subgraph_data.tsv", 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('"', '#%#')

    with open("../dataset/subgraph_data_fix.tsv", 'w') as file:
      file.write(filedata)

    df = pd.read_csv("../dataset/subgraph_data_fix.tsv", sep='\t', encoding='ISO-8859-1')

    df = df.drop_duplicates()
    df1 = df.query('p != "<http://www.disit.org/km4city/schema#eventCategory>"')
    df2 = df.query('p == "<http://www.disit.org/km4city/schema#eventCategory>"')

    df1.to_csv('../dataset/km4city/dataset_for_node_classification/classification/km4c_stripped.nt',index=False, sep=' ')

    df2 = df2[~df2['o'].str.contains('@en')]
    df2.reset_index(drop=True, inplace=True)
    df2.drop('p', inplace=True, axis=1)
    df2.rename(columns={'s': 'nodes',
                       'o': 'label'},
              inplace=True, errors='raise')
    df2.index = np.arange(1, len(df2)+1)
    df2.index.name = 'id'
    df2.to_csv('../dataset/km4city/dataset_for_node_classification/classification/completeDataset.tsv', index=True, sep='\t')


    df = pd.read_csv("../dataset/km4city/dataset_for_node_classification/classification/completeDataset.tsv", sep='\t', encoding='ISO-8859-1')
    df3 = df[['nodes', 'id', 'label']]
    df3.index = np.arange(1, len(df3)+1)
    df3.to_csv('../dataset/km4city/dataset_for_node_classification/classification/daSplittare.tsv', index=False, sep='\t')


    file_name = '../dataset/km4city/dataset_for_node_classification/classification/km4c_stripped.nt'
    string_to_add = " ."

    with open(file_name, 'r',encoding='ISO-8859-1') as f:
        file_lines = [''.join([x.strip(), string_to_add, '\n']) for x in f.readlines()]

    with open(file_name, 'w', encoding='ISO-8859-1') as f:
        f.writelines(file_lines)

    fix_quotation('../dataset/km4city/dataset_for_node_classification/classification/completeDataset.tsv', True)

    with open('../dataset/km4city/dataset_for_node_classification/classification/km4c_stripped.nt', 'r', encoding="ISO-8859-1") as file:
        filedata = file.read()

    filedata = filedata.replace('s p o .\n', '')
    filedata = filedata.replace('"', '')
    filedata = filedata.replace('#%#', '"')

    with open('../dataset/km4city/dataset_for_node_classification/classification/km4c_stripped.nt', 'w', encoding="ISO-8859-1") as file:
        file.write(filedata)

    input = open('../dataset/km4city/dataset_for_node_classification/classification/km4c_stripped.nt', 'rb')
    s = input.read()
    input.close()

    output = gzip.GzipFile('../dataset/km4city/dataset_for_node_classification/classification/km4c_stripped.nt.gz', 'wb')
    output.write(s)
    output.close()



if __name__ == "__main__":
    reformat_data()