import pandas as pd
import numpy as np
import csv
import gzip
from sklearn.model_selection import train_test_split
import re

def fix_quotation(file_path, remove_brakets=False):
    with open(file_path, 'r', encoding="ISO-8859-1") as file:
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

    with open(file_path, 'w', encoding="ISO-8859-1") as file:
        file.write(filedata)

    with open(file_path, 'r',encoding="iso-8859-1") as fr:
        with open(file_path.replace("XXX",""), 'w',encoding='UTF-8') as fw:
            for line in fr:
                fw.write(line[:-1] + '\n')


def reformat_data():
    with open("../dataset/subgraph_data.tsv", 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('"', 'DIOCANE')

    with open("../dataset/subgraph_data_fix.tsv", 'w') as file:
      file.write(filedata)

    df = pd.read_csv("../dataset/subgraph_data_fix.tsv", sep='\t', encoding='ISO-8859-1')

    df = df.drop_duplicates()
    df = df.replace(r'[^0-9a-zA-Z ]', '', regex=True)

    train_val, test = train_test_split(df, test_size=0.2, shuffle=False)
    train, val = train_test_split(train_val, test_size=0.2, shuffle=False)

    train.to_csv('../dataset/km4city/dataset_for_link_prediction/classification/trainXXX.txt', index=False, sep='\t')
    test.to_csv('../dataset/km4city/dataset_for_link_prediction/classification/testXXX.txt', index=False, sep='\t')
    val.to_csv('../dataset/km4city/dataset_for_link_prediction/classification/validXXX.txt', index=False, sep='\t')

    fix_quotation('../dataset/km4city/dataset_for_link_prediction/classification/testXXX.txt')
    fix_quotation('../dataset/km4city/dataset_for_link_prediction/classification/trainXXX.txt')
    fix_quotation('../dataset/km4city/dataset_for_link_prediction/classification/validXXX.txt')


if __name__ == "__main__":
    reformat_data()