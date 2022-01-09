"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn
Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import time
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv
from dgl.data.rdf import BGSDataset


class RGCN(nn.Module):
    def __init__(self, num_nodes, emb_dim, out_dim, num_rels,
                 num_hidden_layers=1):
        super(RGCN, self).__init__() # inizializzazione di default
        self.layers = nn.ModuleList() # oggetto che incapsula i layer
        self.num_nodes = num_nodes # numero di entità
        self.emb_dim = emb_dim  # iperparametro
        self.out_dim = out_dim  # numero di entità da classificare
        self.num_rels = num_rels # numero di tipi di relazioni
        self.num_hidden_layers = num_hidden_layers  # iperparametro
        self.use_self_loop = False  # TODO CHECK va scelto nel caso in cui si faccia link o node classification
        # create rgcn layers
        self.build_model()

    def build_model(self):
        # i2h
        self.layers.append(self.build_input_layer())
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(self.build_hidden_layer())
        # h2o
        self.layers.append(self.build_output_layer())

    def create_features(self):
        features = torch.arange(self.num_nodes)  # (0, 1,___, n-1)
        return features

    def build_input_layer(self):
        return RelGraphConv(self.num_nodes, self.emb_dim, self.num_rels, activation=F.relu,
                            self_loop=self.use_self_loop)

    def build_hidden_layer(self):
        return RelGraphConv(self.emb_dim, self.emb_dim, self.num_rels, activation=F.relu, self_loop=self.use_self_loop)

    def build_output_layer(self):
        return RelGraphConv(self.emb_dim, self.out_dim, self.num_rels, activation=None, self_loop=self.use_self_loop)

    # TODO CHECK va implementata per forza
    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


def main(args):
    print('\n\n----- Dataset:')
    dataset = BGSDataset()  # dataset di default
    print(dataset)

    # Load from hetero-graph
    print('\n\n----- Hetero-Graph:')
    kg = dataset[0]  # knoledge graph? Hetero Graph
    print(kg)

    print('\n\n----- Numero di tipi di relazioni:')
    num_rels = len(kg.canonical_etypes)  # numero di tipi di relazioni
    print(num_rels)

    print('\n\n----- Entità da classificare???????:')
    category = dataset.predict_category # categoria da classificare???
    print(category)

    print('\n\n----- Numero delle classi:')
    num_classes = dataset.num_classes # numero delle classi
    print(num_classes)

    print('\n\n----- Train Mask:')
    train_mask = kg.nodes[category].data.pop('train_mask')
    print(train_mask)

    test_mask = kg.nodes[category].data.pop('test_mask')

    print('\n\n----- Train idx:')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    print(train_idx)

    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()


    labels = kg.nodes[category].data.pop('labels')

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # calculate norm for each edge type and store in edge
    for canonical_etype in kg.canonical_etypes:
        u, v, eid = kg.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = torch.ones(eid.shape[0]).float() / degrees.float()
        norm = norm.unsqueeze(1)
        kg.edges[canonical_etype].data['norm'] = norm

    # get target category id
    category_id = len(kg.ntypes)
    for i, ntype in enumerate(kg.ntypes):
        if ntype == category:
            category_id = i

    g = dgl.to_homogeneous(kg, edata=['norm'])
    num_nodes = g.number_of_nodes()
    node_ids = torch.arange(num_nodes)
    edge_norm = g.edata['norm']
    edge_type = g.edata[dgl.ETYPE].long()

    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]

    # since the nodes are featureless, the input feature is then the node id.
    feats = torch.arange(num_nodes)

    # create model
    model = RGCN(num_nodes=num_nodes,
                 emb_dim=args.n_hidden,
                 out_dim=num_classes,
                 num_rels=num_rels,
                 num_hidden_layers=args.n_layers - 2)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g, feats, edge_type, edge_norm)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
    print()

    model.eval()
    logits = model.forward(g, feats, edge_type, edge_norm)
    logits = logits[target_idx]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))

    # print()
    # print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    # print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--n-hidden", type=int, default=3,
                        help="number of hidden units") # TODO CHECK XXX se questa è emb (prima def era 16)
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    # parser.add_argument("--n-bases", type=int, default=-1,
    #                     help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=15,
                        help="number of training epochs")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
