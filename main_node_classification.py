import os
import manage_dataset.loader as ll
from gnn.node_classification.model import *

# XXX old method used to load data from server
# load graph data
# from dgl.contrib.data import load_data
# data = load_data(dataset='aifb')

# TODO RANDOM SPLIT DATASET
datasets = 'km4c'
curr_dir = str(os.getcwd()) + "\\dataset\\km4city\\dataset_for_node_classification\\classification"
data = ll.load_dataset(label_header='label', nodes_header='nodes', datasets=datasets, dir=curr_dir)

# parametri
n_hidden = 16
num_hidden_layers = 1
epochs = 1
l2norm = 0.0005
lr = 0.01
num_bases = 28

model = Model(data=data, h_dim=n_hidden, num_hidden_layers=num_hidden_layers, n_epochs=epochs, l2norm=l2norm,
              lr=lr, num_bases=num_bases)

model.fit()

model.predict()

entities_vector = [1, 2, 3]
model.predict_single(entities_vector)

# TODO booleano che elimina: edges, labels, test_idx, train_idx