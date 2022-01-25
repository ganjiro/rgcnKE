from model import *
import os
import time
import loader as ll
import dgl
import torch
import numpy as np

# load graph data
from dgl.contrib.data import load_data
data = load_data(dataset='aifb')
datasets = 'aifb'
curr_dir = str(os.getcwd()) + str('\\') + str(datasets) + str('\\')
print("we", curr_dir)
# data = ll.load_dataset(label_header='label', nodes_header='nodes', datasets=datasets, dir=curr_dir)

num_nodes = data.num_nodes
num_rels = data.num_rels
num_classes = data.num_classes
labels = data.labels
train_idx = data.train_idx
# split training and validation set
val_idx = train_idx[:len(train_idx) // 5]
train_idx = train_idx[len(train_idx) // 5:]

# added
test_idx = data.test_idx
# ----------

# edge type and normalization factor
edge_type = torch.from_numpy(data.edge_type)
edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)

labels = torch.from_numpy(labels).view(-1)

# configurations
n_hidden = 16  # number of hidden units
n_bases = -1  # use number of relations as number of bases
n_hidden_layers = 0  # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 25  # epochs to train
lr = 0.01  # learning rate
l2norm = 0  # L2 norm coefficient

# create graph
g = dgl.graph((data.edge_src, data.edge_dst))
g.edata.update({'rel_type': edge_type, 'norm': edge_norm})

# XXX LONG num_nodes, len(g), g.number_of_nodes
model = Model(8285,
              n_hidden,
              num_classes,
              num_rels,
              num_bases=n_bases,
              num_hidden_layers=n_hidden_layers)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

print("start training...")
model.train()

# --------
forward_time = []
backward_time = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    t0 = time.time()
    logits = model.forward(g)
    loss = F.cross_entropy(logits[train_idx], labels[train_idx].long())  # XXX type checl
    t1 = time.time()
    loss.backward()
    optimizer.step()
    t2 = time.time()
    forward_time.append(t1 - t0)
    backward_time.append(t2 - t1)

    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
    train_acc = train_acc.item() / len(train_idx)
    val_loss = F.cross_entropy(logits[val_idx], labels[val_idx].long())  # XXX type check
    val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx])
    val_acc = val_acc.item() / len(val_idx)
    print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
          format(epoch, forward_time[-1], backward_time[-1]))
    print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
          format(train_acc, loss.item(), val_acc, val_loss.item()))

print()

model.eval()
logits = model.forward(g)
test_loss = F.cross_entropy(logits[test_idx], labels[test_idx].long())
test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
print()

print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))
