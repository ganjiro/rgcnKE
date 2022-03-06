# TODO cambiare EXXX con l'URI e abbiamo finito

import configparser
import os

import src.RGCN.loader as ll
from src.DataManagement.reformat_data import split_dataset, reformat_data
from src.RGCN.model import *


class RGCN_node_classification():

    def __init__(self, directory, dataset='km4city'):
        self.dataset = dataset
        self.directory = directory
        config = configparser.ConfigParser()
        config.read(r'{}\parameters\RGCN.ini'.format(self.directory.replace('dataset', "src")))
        self.random_split = config.getboolean('config', 'random_split')
        self.n_hidden = config.getint('config', 'n_hidden')
        self.num_hidden_layers = config.getint('config', 'num_hidden_layers')
        self.epochs = config.getint('config', 'epochs')
        self.l2norm = config.getfloat('config', 'l2norm')
        self.lr = config.getfloat('config', 'lr')
        self.num_bases = config.getint('config', 'num_bases')

    def fit(self):
        print("************** Launching rgcn fit... ************************")

        if self.random_split or not os.path.exists(
                r"{}/{}/NodeClassification/RGCN/train.txt".format(self.directory, self.dataset)):
            split_dataset(r"{}/{}/NodeClassification/RGCN/unsplitted.tsv".format(self.directory, self.dataset),
                          test_size=0.2)
        self.data = ll.load_dataset(label_header='label', nodes_header='nodes', datasets=self.dataset,
                                    dir=r"{}/{}/NodeClassification/RGCN/".format(self.directory, self.dataset))

        self.model = Model(data=self.data, h_dim=self.n_hidden, num_hidden_layers=self.num_hidden_layers,
                           n_epochs=self.epochs, l2norm=self.l2norm,
                           lr=self.lr, num_bases=self.num_bases, directory=self.directory)
        self.model.fit()
        print("************** ending ************************\n\n\n")


    def predict(self):
        self.model.predict()

    def predict_single(self, entities_vector):
        self.model.predict_single(entities_vector)


# if __name__ == '__main__':
#     reformat_data(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset", "km4city", data_type="node")
#     RGCN_node_classification(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset").fit()
