# TODO incapsulare un po' il main e abbiamo finito
# TODO salvare le predizioni

import rdflib
from sklearn.metrics import accuracy_score
from comparing_model.node_classification_MINDWALC.datastructures import Graph
from comparing_model.node_classification_MINDWALC.tree_builder import MINDWALCTree
from Src.DataManagement.reformat_data import split_dataset, reformat_data
import os
import pandas as pd

# labels = np.load("km4city/labels.npz")
# labels = labels["indices"]

class MINDWALC_node_classification():

    def __init__(self, directory, dataset='km4city'):
        self.dataset = dataset
        self.directory = directory
        self.random_split = False  # todo caricarlo da ini
        self.entity_col = 'nodes'
        self.label_col = 'label'
        self.label_predicates = [
            rdflib.URIRef('http://www.disit.org/km4city/schema#eventCategory')
        ]

    def fit(self):
        if self.random_split or not os.path.exists(r"{}/{}/NodeClassification/MINDWALC/train.txt".format(self.directory, self.dataset)) :
            split_dataset(r"{}/{}/NodeClassification/MINDWALC/unsplitted.tsv".format(self.directory, self.dataset),test_size=0.2)

        rdf_file = r"{}/{}/NodeClassification/MINDWALC/{}_stripped.nt".format(self.directory, self.dataset, self.dataset)

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

        test_entities = [rdflib.URIRef(x) for x in test_data[self.entity_col]]
        test_labels = test_data[self.label_col]

        kg = Graph.rdflib_to_graph(g, label_predicates=self.label_predicates)

        self.model = MINDWALCTree()

        self.model.fit(kg, train_entities, train_labels)
    def predict(self):
        preds = self.model.predict(kg, test_entities, test_labels)
        print(accuracy_score(test_labels, preds))
        return preds

    # print("Entità   Classe-Reale    Classe-Predetta")
    # for i in range(len(preds)):
    #     print("E" + str(test_entities[i]) + ":  " + str(test_labels[i]) +
    #           " " + str(preds[i]))

if __name__=='__main__':
    reformat_data(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset","km4city",data_type="node")
    MINDWALC_node_classification(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset").fit()