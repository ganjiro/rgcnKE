import rdflib
import pandas as pd
from sklearn.metrics import accuracy_score

import numpy as np

# labels = np.load("km4city/labels.npz")
# labels = labels["indices"]

from gnn.link_prediction.tree_builder import MINDWALCTree, MINDWALCForest, MINDWALCTransform
from gnn.link_prediction.datastructures import Graph

rdf_file = "dataset\\km4city\\dataset_for_link_prediction\\classification\\km_link_pred.nt"
_format = 'nt'
train_file = 'dataset\\km4city\\dataset_for_link_prediction\\classification\\trainingSet.tsv'
test_file = 'dataset\\km4city\\dataset_for_link_prediction\\classification\\testSet.tsv'
entity_col = 'nodes'
label_col = 'label'
label_predicates = [
    rdflib.URIRef('http://www.disit.org/km4city/schema#eventCategory')
]
output = 'dataset\\km4city\\dataset_for_link_prediction\\output\\link_results.p'

print(end='Loading data... ', flush=True)
g = rdflib.Graph()
g.parse(rdf_file, format=_format)
print('OK')

test_data = pd.read_csv(train_file, sep='\t')
train_data = pd.read_csv(test_file, sep='\t')

train_entities = [rdflib.URIRef(x) for x in train_data[entity_col]]
train_labels = train_data[label_col]

test_entities = [rdflib.URIRef(x) for x in test_data[entity_col]]
test_labels = test_data[label_col]

kg = Graph.rdflib_to_graph(g, label_predicates=label_predicates)

# à###########hediwuwhduw

clf = MINDWALCTree()
# clf = MINDWALCForest()
# clf = MINDWALCTransform()

clf.fit(kg, train_entities, train_labels)

preds = clf.predict(kg, test_entities)
print(accuracy_score(test_labels, preds))
