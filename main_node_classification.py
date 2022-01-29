import os
import loader as ll
from gnn.node_classification.model import *

# XXX old method used to load data from server
# load graph data
# from dgl.contrib.data import load_data
# data = load_data(dataset='aifb')

datasets = 'km4c'
curr_dir = str(os.getcwd()) + "\\km4city\\"
data = ll.load_dataset(label_header='label', nodes_header='nodes', datasets=datasets, dir=curr_dir)

# parametri
n_hidden = 20
num_hidden_layers = 1
epochs = 10000
l2norm = 5e-4
lr = 0.04
num_bases = 16

model = Model(data=data, h_dim=n_hidden, num_hidden_layers=num_hidden_layers, n_epochs=epochs, l2norm=l2norm,
              lr=lr, num_bases=num_bases)

model.fit()
