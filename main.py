import os
import loader as ll
from gnn.model import *

# XXX old method used to load data from server
# load graph data
# from dgl.contrib.data import load_data
# data = load_data(dataset='aifb')

datasets = 'km4c'
curr_dir = str(os.getcwd())+"\\km4city\\"
data = ll.load_dataset(label_header='label', nodes_header='nodes', datasets=datasets, dir=curr_dir)

# parametri
# configurations
n_hidden = 16  # number of hidden units

model = Model(data=data, h_dim=n_hidden)

model.fit()

