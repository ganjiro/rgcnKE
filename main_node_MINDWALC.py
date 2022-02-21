# TODO incapsulare un po' il main e abbiamo finito
# TODO salvare le predizioni

import rdflib
from sklearn.metrics import accuracy_score

from comparing_model.node_classification_MINDWALC.datastructures import Graph
from comparing_model.node_classification_MINDWALC.tree_builder import MINDWALCTree
from manage_dataset.splidataset import *

# labels = np.load("km4city/labels.npz")
# labels = labels["indices"]


path = str(os.getcwd()) + "\\dataset\\km4city\\dataset_for_node_classification\\classification"
prev_path = str(os.getcwd()) + "\\dataset\\km4city\\dataset_for_node_classificatio"

split_dataset(test_size=0.2, path=path)
rdf_file = str(path) + "\\km4c_stripped.nt"
_format = 'nt'
train_file = str(path) + "\\trainingSet.tsv"
test_file = str(path) + "\\testSet.tsv"
entity_col = 'nodes'
label_col = 'label'
label_predicates = [
    rdflib.URIRef('http://www.disit.org/km4city/schema#eventCategory')
]
output = str(prev_path) + "\\output\\node_MINDWALC_results.p"

print(end='Loading data... ', flush=True)
g = rdflib.Graph()
g.parse(rdf_file, format=_format)
print('OK')
print()

test_data = pd.read_csv(train_file, sep='\t')
train_data = pd.read_csv(test_file, sep='\t')

train_entities = [rdflib.URIRef(x) for x in train_data[entity_col]]
train_labels = train_data[label_col]

test_entities = [rdflib.URIRef(x) for x in test_data[entity_col]]
test_labels = test_data[label_col]

kg = Graph.rdflib_to_graph(g, label_predicates=label_predicates)

clf = MINDWALCTree()

clf.fit(kg, train_entities, train_labels)

preds = clf.predict(kg, test_entities, test_labels)

# print("Entità   Classe-Reale    Classe-Predetta")
# for i in range(len(preds)):
#     print("E" + str(test_entities[i]) + ":  " + str(test_labels[i]) +
#           " " + str(preds[i]))
print(accuracy_score(test_labels, preds))
