import configparser
import os

import pandas as pd
import rdflib
from sklearn.metrics import accuracy_score

from src.DataManagement.reformat_data import split_dataset
from src.MINDWALC.datastructures import Graph
from src.MINDWALC.tree_builder import MINDWALCTree


# labels = np.load("km4city/labels.npz")
# labels = labels["indices"]

class MINDWALC_node_classification():

    def __init__(self, directory, dataset='km4city'):
        self.kg = None
        self.dataset = dataset
        self.directory = directory
        self.entity_col = 'nodes'
        self.label_col = 'label'
        self.label_predicates = [
            rdflib.URIRef('http://www.disit.org/km4city/schema#eventCategory')
        ]
        config = configparser.ConfigParser()
        config.read(r'{}\parameters\mindwalc.ini'.format(self.directory.replace('dataset', "src")))

        self.random_split = config['config'].getboolean('random_split')

    def fit(self):
        print("************** Launching Decision Tree fit... ************************")


        if self.random_split or not os.path.exists(
                r"{}/{}/NodeClassification/MINDWALC/train.txt".format(self.directory, self.dataset)):
            split_dataset(r"{}/{}/NodeClassification/MINDWALC/unsplitted.tsv".format(self.directory, self.dataset),
                          test_size=0.2)

        rdf_file = r"{}/{}/NodeClassification/MINDWALC/{}_stripped.nt".format(self.directory, self.dataset,
                                                                              self.dataset)

        train_file = r"{}/{}/NodeClassification/MINDWALC/train.txt".format(self.directory, self.dataset)
        test_file = r"{}/{}/NodeClassification/MINDWALC/test.txt".format(self.directory, self.dataset)

        print(end='Loading data... ', flush=True)
        g = rdflib.Graph()
        g.parse(rdf_file, format='nt')
        print('OK')
        print()

        test_data = pd.read_csv(train_file, sep='\t')
        train_data = pd.read_csv(test_file, sep='\t')

        train_entities = [rdflib.URIRef(x) for x in train_data[self.entity_col]]
        train_labels = train_data[self.label_col]

        self.test_entities = [rdflib.URIRef(x) for x in test_data[self.entity_col]]
        self.test_labels = test_data[self.label_col]

        kg = Graph.rdflib_to_graph(g, label_predicates=self.label_predicates)

        self.model = MINDWALCTree(directory=self.directory)

        self.model.fit(kg, train_entities, train_labels)

        self.kg = kg
        print("************** ending ************************\n\n\n")

    def predict(self):
        preds = self.model.predict(kg=self.kg, instances=self.test_entities, test_labes=self.test_labels)
        print(accuracy_score(self.test_labels, preds))
        return preds

    # print("Entità   Classe-Reale    Classe-Predetta")
    # for i in range(len(preds)):
    #     print("E" + str(test_entities[i]) + ":  " + str(test_labels[i]) +
    #           " " + str(preds[i]))


# if __name__ == '__main__':
#     # reformat_data(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset","km4city",data_type="node")
#     MINDWALC_node_classification(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset").fit()
