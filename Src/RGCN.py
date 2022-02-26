# TODO cambiare EXXX con l'URI e abbiamo finito
# TODO salvare le predizioni
# TODO tuning

import Src.gnn.node_classification.loader as ll
from Src.gnn.node_classification.model import *
from Src.DataManagement.reformat_data import split_dataset, reformat_data


# XXX old method used to load data from server
# load graph data
# from dgl.contrib.data import load_data
# data = load_data(dataset='aifb')

class RGCN_node_classification():

    def __init__(self, directory, dataset='km4city'):
        self.dataset = dataset
        self.directory = directory
        self.random_split = False  # todo caricarlo da ini
        self.n_hidden = 16
        self.num_hidden_layers = 1
        self.epochs = 1
        self.l2norm = 0.0005
        self.lr = 0.01
        self.num_bases = 28

    def fit(self):
        split_dataset(r"{}/{}/NodeClassification/RGCN/unsplitted.tsv".format(self.directory, self.dataset),test_size=0.2 )
        self.data = ll.load_dataset(label_header='label', nodes_header='nodes', datasets=self.dataset,
                                    dir=r"{}/{}/NodeClassification/RGCN/".format(self.directory, self.dataset))

        self.model = Model(data=self.data, h_dim=self.n_hidden, num_hidden_layers=self.num_hidden_layers,
                           n_epochs=self.epochs, l2norm=self.l2norm,
                           lr=self.lr, num_bases=self.num_bases)
        self.model.fit()

    def predict(self):
        self.model.predict()

    def predict_single(self, entities_vector):
        self.model.predict_single(entities_vector)

if __name__=='__main__':
    #reformat_data(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset","km4city",data_type="node")
    RGCN_node_classification(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset").fit()