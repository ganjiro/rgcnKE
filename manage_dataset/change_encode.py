
import os

with open("C:\\Users\\Girolamo\\PycharmProjects\\rgcnKE_sus\\dataset\\km4c_stripped.nt", 'r', encoding="iso-8859-1") as fr:
    with open("C:\\Users\\Girolamo\\PycharmProjects\\rgcnKE_sus\dataset\\km_link_pred.nt", 'w', encoding='UTF-8') as fw:
        for line in fr:
            fw.write(line[:-1]+'\n')


