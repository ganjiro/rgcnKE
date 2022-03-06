import gzip
import os
from distutils.dir_util import copy_tree
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def open_secure(path, type, encoding=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if encoding:
        file = open(path, type, encoding=encoding)
    else:
        file = open(path, type)
    return file


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
    filedata = filedata.replace('XZXZX', '"')
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


def reformat_data_for_pykeen(directory, dataset='km4city'):
    with open_secure(r"{}/{}/RawDowloaded/subgraph_data.tsv".format(directory, dataset), 'r',
                     encoding='ISO-8859-1') as file:
        filedata = file.read()

    filedata = filedata.replace('"', 'XZXZX')

    with open_secure(r"{}/{}/RawDowloaded/subgraph_data_fix.tsv".format(directory, dataset), 'w',
                     encoding='ISO-8859-1') as file:
        file.write(filedata)

    df = pd.read_csv(r"{}/{}/RawDowloaded/subgraph_data_fix.tsv".format(directory, dataset), sep='\t',
                     encoding='ISO-8859-1')

    df = df.drop_duplicates()
    df = df.replace(r'[^0-9a-zA-Z ]', '', regex=True)

    df.to_csv(r'{}/{}/linkPrediction/Pykeen/CompleteXXX.txt'.format(directory, dataset), index=False, sep='\t')
    fix_quotation_link(r'{}/{}/linkPrediction/Pykeen/CompleteXXX.txt'.format(directory, dataset))

    if os.path.exists(r'{}/{}/linkPrediction/Pykeen/CompleteXXX.txt'.format(directory, dataset)):
        os.remove(r'{}/{}/linkPrediction/Pykeen/CompleteXXX.txt'.format(directory, dataset))

    if os.path.exists(r'{}/{}/RawDowloaded/subgraph_data_fix.tsv'.format(directory, dataset)):
        os.remove(r'{}/{}/RawDowloaded/subgraph_data_fix.tsv'.format(directory, dataset))


def reformat_data_for_noge(directory, dataset='km4city'):
    with open_secure(r"{}/{}/RawDowloaded/subgraph_data.tsv".format(directory, dataset), 'r',
                     encoding='ISO-8859-1') as file:
        filedata = file.read()

    filedata = filedata.replace('"', 'XZXZX')

    with open_secure(r"{}/{}/RawDowloaded/subgraph_data_fix.tsv".format(directory, dataset), 'w',
                     encoding='ISO-8859-1') as file:
        file.write(filedata)

    df = pd.read_csv(r"{}/{}/RawDowloaded/subgraph_data_fix.tsv".format(directory, dataset), sep='\t',
                     encoding='ISO-8859-1')

    df = df.drop_duplicates()
    df = df.replace(r'[^0-9a-zA-Z ]', '', regex=True)

    train_val, test = train_test_split(df, test_size=0.2, shuffle=False)
    train, val = train_test_split(train_val, test_size=0.2, shuffle=False)

    train.to_csv(r'{}/{}/linkPrediction/NOGE/trainXXX.txt'.format(directory, dataset), index=False, sep='\t')
    test.to_csv(r'{}/{}/linkPrediction/NOGE/testXXX.txt'.format(directory, dataset), index=False, sep='\t')
    val.to_csv(r'{}/{}/linkPrediction/NOGE/validXXX.txt'.format(directory, dataset), index=False, sep='\t')

    fix_quotation_link(r'{}/{}/linkPrediction/NOGE/testXXX.txt'.format(directory, dataset))
    fix_quotation_link(r'{}/{}/linkPrediction/NOGE/trainXXX.txt'.format(directory, dataset))
    fix_quotation_link(r'{}/{}/linkPrediction/NOGE/validXXX.txt'.format(directory, dataset))


def reformat_data_for_node_classification(directory, dataset='km4city'):
    with open_secure(r"{}/{}/RawDowloaded/subgraph_data.tsv".format(directory, dataset), 'r',
                     encoding='ISO-8859-1') as file:
        filedata = file.read()

    filedata = filedata.replace('"', '#%#')

    with open_secure(r"{}/{}/RawDowloaded/subgraph_data_fix.tsv".format(directory, dataset), 'w',
                     encoding='ISO-8859-1') as file:
        file.write(filedata)

    df = pd.read_csv(r"{}/{}/RawDowloaded/subgraph_data_fix.tsv".format(directory, dataset), sep='\t',
                     encoding='ISO-8859-1')

    df = df.drop_duplicates()
    df1 = df.query('p != "<http://www.disit.org/km4city/schema#eventCategory>"')
    df2 = df.query('p == "<http://www.disit.org/km4city/schema#eventCategory>"')

    df1.to_csv(r'{}/{}/NodeClassification/Rgcn/{}_stripped.nt'.format(directory, dataset, dataset), index=False,
               sep=' ')

    df2 = df2[~df2['o'].str.contains('@en')]
    df2.reset_index(drop=True, inplace=True)
    df2.drop('p', inplace=True, axis=1)
    df2.rename(columns={'s': 'nodes',
                        'o': 'label'},
               inplace=True, errors='raise')
    df2.index = np.arange(1, len(df2) + 1)
    df2.index.name = 'id'
    df2.to_csv(r'{}/{}/NodeClassification/Rgcn/completeDataset.tsv'.format(directory, dataset), index=True, sep='\t')

    df = pd.read_csv(r"{}/{}/NodeClassification/RGCN/completeDataset.tsv".format(directory, dataset), sep='\t',
                     encoding='ISO-8859-1')
    df3 = df[['nodes', 'id', 'label']]
    df3.index = np.arange(1, len(df3) + 1)
    df3.to_csv(r'{}/{}/NodeClassification/Rgcn/unsplitted.tsv'.format(directory, dataset), index=False, sep='\t')

    file_name = r'{}/{}/NodeClassification/Rgcn/{}_stripped.nt'.format(directory, dataset, dataset)
    string_to_add = " ."

    with open_secure(file_name, 'r', encoding='ISO-8859-1') as f:
        file_lines = [''.join([x.strip(), string_to_add, '\n']) for x in f.readlines()]

    with open_secure(file_name, 'w', encoding='ISO-8859-1') as f:
        f.writelines(file_lines)

    fix_quotation_node(r'{}/{}/NodeClassification/RGCN/completeDataset.tsv'.format(directory, dataset), True)

    with open_secure(r'{}/{}/NodeClassification/Rgcn/{}_stripped.nt'.format(directory, dataset, dataset), 'r',
                     encoding="ISO-8859-1") as file:
        filedata = file.read()

    filedata = filedata.replace('s p o .\n', '')
    filedata = filedata.replace('"', '')
    filedata = filedata.replace('#%#', '"')

    with open_secure(r'{}/{}/NodeClassification/Rgcn/{}_stripped.nt'.format(directory, dataset, dataset), 'w',
                     encoding="ISO-8859-1") as file:
        file.write(filedata)

    input = open_secure(r'{}/{}/NodeClassification/Rgcn/{}_stripped.nt'.format(directory, dataset, dataset), 'rb')
    s = input.read()
    input.close()

    output = gzip.GzipFile(r'{}/{}/NodeClassification/Rgcn/{}_stripped.nt.gz'.format(directory, dataset, dataset), 'wb')
    output.write(s)
    output.close()

    copy_tree(r'{}/{}/NodeClassification/Rgcn/'.format(directory, dataset),
              r'{}/{}/NodeClassification/MINDWALC/'.format(directory, dataset))
    filelist = [f for f in os.listdir(os.path.join(r'{}/{}/NodeClassification/Rgcn/'.format(directory, dataset))) if
                f.endswith(".npz") or f.endswith(".npy")]
    for f in filelist:
        os.remove(os.path.join(r'{}/{}/NodeClassification/Rgcn/'.format(directory, dataset), f))

    if os.path.exists(r"{}/{}/RawDowloaded/subgraph_data_fix.tsv".format(directory, dataset)):
        os.remove(r"{}/{}/RawDowloaded/subgraph_data_fix.tsv".format(directory, dataset))


def split_dataset(filename, test_size=0.2, node=True):
    seed = randint(1, 1000)
    df = pd.read_csv(filename, sep='\t', encoding='ISO-8859-1')
    train, test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    train.to_csv(r'{}\test.txt'.format(os.path.dirname(filename)), index=False, sep='\t')
    test.to_csv(r'{}\train.txt'.format(os.path.dirname(filename)), index=False, sep='\t')
    if node:
        fix_quotation_node(r'{}\test.txt'.format(os.path.dirname(filename)), True)
        fix_quotation_node(r'{}\train.txt'.format(os.path.dirname(filename)), True)

        if os.path.isfile(r'{}\edges.npz'.format(os.path.dirname(filename))):
            os.remove(r'{}\edges.npz'.format(os.path.dirname(filename)))
    else:
        fix_quotation_link(r'{}\test.txt'.format(os.path.dirname(filename)), True)
        fix_quotation_link(r'{}\train.txt'.format(os.path.dirname(filename)), True)


def to_utf8(filename_in):
    with open_secure(filename_in, 'r', encoding="iso-8859-1") as fr:
        with open_secure(r'{}utf'.format(filename_in), 'w', encoding='UTF-8') as fw:
            for line in fr:
                fw.write(line[:-1] + '\n')
    os.remove(filename_in)
    os.rename(r'{}utf'.format(filename_in), filename_in)


def reformat_data(directory, dataset='km4city', data_type="node"):
    to_utf8(r"{}/{}/RawDowloaded/subgraph_data.tsv".format(directory, dataset))
    if data_type.lower() == "node":
        reformat_data_for_node_classification(directory, dataset)
    elif data_type.lower() == "noge":
        reformat_data_for_noge(directory, dataset)
    elif data_type.lower() == "pykeen":
        reformat_data_for_pykeen(directory, dataset)
    else:
        raise Exception("Model not found")


if __name__ == "__main__":
    reformat_data(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset", 'km4city', data_type="pykeen")
