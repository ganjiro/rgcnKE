import time
from functools import partial
import dgl
import torch
import numpy as np
import dgl
import numpy as np

from gnn.layer import *
import torch.nn.functional as F


class RGCN(nn.Module):
    def __init__(self, data, h_dim, num_hidden_layers=1, num_bases=-1, lr=1e-3, n_epochs = 30):
        super(RGCN, self).__init__()
        self.data = data
        self.lr = lr # serve per l'otttimizzatore
        self.n_epochs = n_epochs # serve nel fit
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.labels = None
        self.train_idx = None
        self.test_idx = None
        self.val_idx = None
        self.num_nodes = None
        self.num_rels = None  # nel prepare
        self.h_dim = None  # nel prepare
        self.out_dim = None # nel prepare
        self.graph = self.prepare_graph()
        self.h_dim = h_dim
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()
        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')

    def prepare_graph(self):
        num_nodes = self.data.num_nodes
        self.num_rels = self.data.num_rels
        self.out_dim = self.data.num_classes
        self.labels = self.data.labels
        self.train_idx = self.data.train_idx
        # split training and validation set
        self.val_idx = self.train_idx[:len(self.train_idx) // 5]
        self.train_idx = self.train_idx[len(self.train_idx) // 5:]
        # added
        self.test_idx = self.data.test_idx
        # ----------

        # edge type and normalization factor
        edge_type = torch.from_numpy(self.data.edge_type)
        edge_norm = torch.from_numpy(self.data.edge_norm).unsqueeze(1)

        self.labels = torch.from_numpy(self.labels).view(-1)

        # create graph
        g = dgl.graph((self.data.edge_src, self.data.edge_dst))
        g.edata.update({'rel_type': edge_type, 'norm': edge_norm})
        self.num_nodes = 60373 # TODO FIX
        return g

    def fit(self, want_accuracy = True):
        print("start training...")
        self.train()

        # --------
        forward_time = []
        backward_time = []

        for epoch in range(self.n_epochs):
            # zero_grad deprecated: instead: cleargrads()
            self.optimizer.zero_grad()
            t0 = time.time()
            logits = self.forward(self.graph)
            loss = F.cross_entropy(logits[self.train_idx], self.labels[self.train_idx].long())  # XXX type checl
            t1 = time.time()
            loss.backward()
            self.optimizer.step()
            t2 = time.time()
            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)

            train_acc = torch.sum(logits[self.train_idx].argmax(dim=1) == self.labels[self.train_idx])
            train_acc = train_acc.item() / len(self.train_idx)
            val_loss = F.cross_entropy(logits[self.val_idx], self.labels[self.val_idx].long())  # XXX type check
            val_acc = torch.sum(logits[self.val_idx].argmax(dim=1) == self.labels[self.val_idx])
            val_acc = val_acc.item() / len(self.val_idx)
            print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
                  format(epoch, forward_time[-1], backward_time[-1]))
            print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
                  format(train_acc, loss.item(), val_acc, val_loss.item()))

        print()
        if want_accuracy:
            self.print_accuracy(forward_time, backward_time)

    def print_accuracy(self, forward_time, backward_time):
        # optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.eval()
        logits = self.forward(self.graph)
        test_loss = F.cross_entropy(logits[self.test_idx], self.labels[self.test_idx].long())
        test_acc = torch.sum(logits[self.test_idx].argmax(dim=1) == self.labels[self.test_idx]).item() / len(self.test_idx)
        print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
        print()

        print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
        print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))