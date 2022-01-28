import os
import loader as ll
from gnn.rgcn import *

# XXX old method used to load data from server
# load graph data
# from dgl.contrib.data import load_data
# data = load_data(dataset='aifb')

datasets = 'km4c'
curr_dir = str(os.getcwd())+'\\km4city\\'
data = ll.load_dataset(label_header='label', nodes_header='nodes', datasets=datasets, dir=curr_dir)

# parametri
n_hidden = 16  # number of hidden units
n_bases = -1  # use number of relations as number of bases
n_hidden_layers = 1  # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 100000  # epochs to train
# lr = 0.01  # learning rate
# l2norm = 0  # L2 norm coefficient


model = RGCN(data = data, h_dim=n_hidden, num_bases=n_bases, num_hidden_layers=n_hidden_layers)

model.fit()

model.print_accuracy()
