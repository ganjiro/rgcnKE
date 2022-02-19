
import os

with open("C:\\Users\\mistr\\OneDrive\\Desktop\\pythonProject\\GNNforRDFs\\dataset\\km4city\\dataset_for_link_prediction"
          "\\preparation\\km_link_pred.nt", 'r', encoding="iso-8859-1") as fr:
    with open("C:\\Users\\mistr\\OneDrive\\Desktop\\pythonProject\\GNNforRDFs\\dataset\\km4city\\dataset_for_link_prediction"
          "\\classification\\km_link_pred.nt", 'w', encoding='UTF-8') as fw:
        for line in fr:
            fw.write(line[:-1]+'\n')


