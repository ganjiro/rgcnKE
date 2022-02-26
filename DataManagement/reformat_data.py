import pandas as pd
import numpy as np
import csv
import gzip
from sklearn.model_selection import train_test_split
import re
import os
from random import randint
from sklearn.model_selection import train_test_split
from pathlib import Path

def open_secure(path, type, encoding = "UTF-8"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, type, encoding=encoding)

def fix_quotation_node(file_path, remove_brakets=False):
    with open_secure(file_path, 'r', encoding="ISO-8859-1") as file:
        filedata = file.read()

    filedata = filedata.replace('#%#', '"')
    if remove_brakets:
        filedata = filedata.replace('<', '')
        filedata = filedata.replace('>', '')

    with open_secure(file_path, 'w', encoding="ISO-8859-1") as file:
        file.write(filedata)

def fix_quotation_link(file_path, remove_brakets=False):
    with open_secure(file_path, 'r', encoding="ISO-8859-1") as file:
        filedata = file.read()

    filedata = filedata.replace('s\tp\to\n', '')
    filedata = filedata.replace('DIOCANE', '"')
    filedata = filedata.replace(' ', '/')
    # filedata = filedata.replace(',', '')
    # filedata = filedata.replace('.', '')
    # filedata = filedata.replace(';', '')
    # filedata = filedata.replace(':', '')
    # filedata = filedata.replace("'", '')
    # filedata = filedata.replace('¨', '')
    # filedata = re.sub('[^A-Za-z0-9]+', '', filedata)
    # filedata = filedata.replace('Ã', '')
    # filedata = filedata.replace('Â', '')
    # filedata = filedata.replace('-', '')
    filedata = filedata.replace('""', 'empty')
    filedata = filedata.replace('"', '')
    # filedata = filedata.replace('@', '')
    # filedata = filedata.replace('/', 'xd')

    if remove_brakets:
        filedata = filedata.replace('<', '')
        filedata = filedata.replace('>', '')

    with open_secure(file_path, 'w', encoding="ISO-8859-1") as file:
        file.write(filedata)

    with open_secure(file_path, 'r', encoding="iso-8859-1") as fr:
        with open_secure(file_path.replace("XXX", ""), 'w', encoding='UTF-8') as fw:
            for line in fr:
                fw.write(line[:-1] + '\n')


def reformat_data_for_pykeen():
    with open_secure("../dataset/km4city/RawDowloaded/subgraph_data.tsv", 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('"', 'XZXZXZX')

    with open_secure("../dataset/km4city/RawDowloaded/subgraph_data_fix.tsv", 'w') as file:
        file.write(filedata)

    df = pd.read_csv("../dataset/km4city/RawDowloaded/subgraph_data_fix.tsv", sep='\t', encoding='ISO-8859-1')

    df = df.drop_duplicates()
    df = df.replace(r'[^0-9a-zA-Z ]', '', regex=True)

    df.to_csv('../dataset/km4city/linkPrediction/Pykeen/CompleteXXX.txt', index=False, sep='\t')
    fix_quotation_link('../dataset/km4city/linkPrediction/Pykeen/CompleteXXX.txt')

    if os.path.exists('../dataset/km4city/linkPrediction/Pykeen/CompleteXXX.txt'):
        os.remove('../dataset/km4city/linkPrediction/Pykeen/CompleteXXX.txt')

    if os.path.exists('../dataset/km4city/RawDowloaded/subgraph_data_fix.tsv'):
        os.remove('../dataset/km4city/RawDowloaded/subgraph_data_fix.tsv')


def reformat_data_for_noge():
    with open_secure("../dataset/km4city/RawDowloaded/subgraph_data.tsv", 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('"', 'DIOCANE')

    with open_secure("../dataset/subgraph_data_fix.tsv", 'w') as file:
        file.write(filedata)

    df = pd.read_csv("../dataset/subgraph_data_fix.tsv", sep='\t', encoding='ISO-8859-1')

    df = df.drop_duplicates()
    df = df.replace(r'[^0-9a-zA-Z ]', '', regex=True)

    train_val, test = train_test_split(df, test_size=0.2, shuffle=False)
    train, val = train_test_split(train_val, test_size=0.2, shuffle=False)

    train.to_csv('../dataset/km4city/dataset_for_link_prediction/classification/trainXXX.txt', index=False, sep='\t')
    test.to_csv('../dataset/km4city/dataset_for_link_prediction/classification/testXXX.txt', index=False, sep='\t')
    val.to_csv('../dataset/km4city/dataset_for_link_prediction/classification/validXXX.txt', index=False, sep='\t')

    fix_quotation_link('../dataset/km4city/dataset_for_link_prediction/classification/testXXX.txt')
    fix_quotation_link('../dataset/km4city/dataset_for_link_prediction/classification/trainXXX.txt')
    fix_quotation_link('../dataset/km4city/dataset_for_link_prediction/classification/validXXX.txt')


def reformat_data_for_rgcn():
    with open_secure("../dataset/km4city/RawDowloaded/subgraph_data.tsv", 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('"', '#%#')

    with open_secure("../dataset/km4city/RawDowloaded/subgraph_data.tsv", 'w') as file:
      file.write(filedata)

    df = pd.read_csv("../dataset/km4city/RawDowloaded/subgraph_data.tsv", sep='\t', encoding='ISO-8859-1')

    df = df.drop_duplicates()
    df1 = df.query('p != "<http://www.disit.org/km4city/schema#eventCategory>"')
    df2 = df.query('p == "<http://www.disit.org/km4city/schema#eventCategory>"')

    df1.to_csv('../dataset/km4city/NodeClassification/Rgcn/km4c_stripped.nt',index=False, sep=' ')

    df2 = df2[~df2['o'].str.contains('@en')]
    df2.reset_index(drop=True, inplace=True)
    df2.drop('p', inplace=True, axis=1)
    df2.rename(columns={'s': 'nodes',
                       'o': 'label'},
              inplace=True, errors='raise')
    df2.index = np.arange(1, len(df2)+1)
    df2.index.name = 'id'
    df2.to_csv('../dataset/km4city/NodeClassification/Rgcn/completeDataset.tsv', index=True, sep='\t')


    df = pd.read_csv("../dataset/km4city/NodeClassification/Rgcn/completeDataset.tsv", sep='\t', encoding='ISO-8859-1')
    df3 = df[['nodes', 'id', 'label']]
    df3.index = np.arange(1, len(df3)+1)
    df3.to_csv('../dataset/km4city/NodeClassification/Rgcn/daSplittare.tsv', index=False, sep='\t')


    file_name = '../dataset/km4city/NodeClassification/Rgcn/km4c_stripped.nt'
    string_to_add = " ."

    with open_secure(file_name, 'r',encoding='ISO-8859-1') as f:
        file_lines = [''.join([x.strip(), string_to_add, '\n']) for x in f.readlines()]

    with open_secure(file_name, 'w', encoding='ISO-8859-1') as f:
        f.writelines(file_lines)

    fix_quotation_node('../dataset/km4city/NodeClassification/Rgcn/completeDataset.tsv', True)

    with open_secure('../dataset/km4city/NodeClassification/Rgcn/km4c_stripped.nt', 'r', encoding="ISO-8859-1") as file:
        filedata = file.read()

    filedata = filedata.replace('s p o .\n', '')
    filedata = filedata.replace('"', '')
    filedata = filedata.replace('#%#', '"')

    with open_secure('../dataset/km4city/NodeClassification/Rgcn/km4c_stripped.nt', 'w', encoding="ISO-8859-1") as file:
        file.write(filedata)

    input = open_secure('../dataset/km4city/NodeClassification/Rgcn/km4c_stripped.nt', 'rb')
    s = input.read()
    input.close()

    output = gzip.GzipFile('../dataset/km4city/NodeClassification/Rgcn/km4c_stripped.nt.gz', 'wb')
    output.write(s)
    output.close()

def split_dataset(filename, test_size=0.2, node = True):
    seed = randint(1, 1000)
    df = pd.read_csv(filename, sep='\t', encoding='ISO-8859-1')
    train, test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    train.to_csv(os.path.dirname + '\\test.txt', index=False, sep='\t')
    test.to_csv(os.path.dirname + '\\train.txt', index=False, sep='\t')
    if node:
        fix_quotation_node(os.path.dirname + '\\test.txt', True)
        fix_quotation_node(os.path.dirname + '\\train.txt', True)

        if os.path.isfile(os.path.dirname + "\\edges.npz"):
            os.remove(os.path.dirname + "\\edges.npz")
    else:
        fix_quotation_link(os.path.dirname + '\\test.txt', True)
        fix_quotation_link(os.path.dirname + '\\train.txt', True)



def to_utf8(filename_in, filename_out):
    with open_secure(filename_in, 'r', encoding="iso-8859-1") as fr:
        with open_secure(filename_out, 'w', encoding='UTF-8') as fw:
            for line in fr:
                fw.write(line[:-1] + '\n')


def reformat_data(data_type="rgcn"):
    if data_type.lower() == "rgcn":
        reformat_data_for_rgcn()
    elif data_type.lower() == "noge":
        reformat_data_for_noge()
    elif data_type.lower() == "pykeen":
        reformat_data_for_pykeen()
    else:
        raise Exception("Model not found")

if __name__ == "__main__":
    reformat_data()