import rdflib
import pandas as pd
from sklearn.metrics import accuracy_score

import numpy as np

# labels = np.load("km4city/labels.npz")
# labels = labels["indices"]


from tree_builder import MINDWALCTree, MINDWALCForest, MINDWALCTransform
from datastructures import Graph


rdf_file = 'km4city/km4c_stripped.nt'
_format = 'nt'
train_file = 'km4city/trainingSet.tsv'
test_file = 'km4city/testSet.tsv'
entity_col = 'nodes'
label_col = 'label'
label_predicates = [
    rdflib.URIRef('http://www.disit.org/km4city/schema#eventCategory')
]
output = 'mio_km4city_compare.p'

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


#à###########hediwuwhduw

clf = MINDWALCTree()
#clf = MINDWALCForest()
#clf = MINDWALCTransform()

clf.fit(kg, train_entities, train_labels)

preds = clf.predict(kg, test_entities)
print(accuracy_score(test_labels, preds))