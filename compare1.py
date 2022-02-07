import rdflib
import pandas as pd
from sklearn.metrics import accuracy_score

from tree_builder import MINDWALCTree, MINDWALCForest, MINDWALCTransform
from datastructures import Graph

g = rdflib.Graph()
g.parse('data/AIFB/aifb.n3', format='n3')

train_data = pd.read_csv('data/AIFB/AIFB_train.tsv', sep='\t')
train_entities = [rdflib.URIRef(x) for x in train_data['person']]
train_labels = train_data['label_affiliation']

test_data = pd.read_csv('data/AIFB/AIFB_test.tsv', sep='\t')
test_entities = [rdflib.URIRef(x) for x in test_data['person']]
test_labels = test_data['label_affiliation']

kg = Graph.rdflib_to_graph(g, label_predicates=label_predicates)

clf = MINDWALCTree()
#clf = MINDWALCForest()
#clf = MINDWALCTransform()

clf.fit(kg, train_entities, train_labels)

preds = clf.predict(kg, test_entities)
print(accuracy_score(test_labels, preds))