import time
import dgl
import numpy as np
from gnn.node_classification.rgcn import *
import torch


class Model(nn.Module):
    def __init__(self, data, h_dim, num_hidden_layers=1, num_bases=51, n_epochs=1000, lr=0.01, l2norm=0.00005):
        super(Model, self).__init__()
        self.data = data
        self.h_dim = h_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_bases = num_bases
        self.n_epochs = n_epochs
        # ----
        self.lr = lr
        self.l2norm = l2norm
        # ----
        self.out_dim = None
        self.num_nodes = None
        self.num_rels = None
        # ----
        self.labels = None
        self.val_idx = None
        self.train_idx = None
        self.test_idx = None

        self.labels_dict = None  # XXX NUOVA AGGIUNTA predict

        # ----
        self.graph = self.prepare_graph()
        self.RGCN = RGCN(num_nodes=self.num_nodes, h_dim=self.h_dim, out_dim=self.out_dim, num_rels=self.num_rels,
                         num_bases=self.num_bases, num_hidden_layers=self.num_hidden_layers)
        self.optimizer = torch.optim.Adam(self.RGCN.parameters(), lr=lr, weight_decay=l2norm)

        self._print_parameters()

    def prepare_graph(self):
        self.labels_dict = self.data.label_dict  # XXX NUOVA AGGIUNTA predict

        # num_nodes = self.data.num_nodes
        self.num_rels = self.data.num_rels
        self.out_dim = self.data.num_classes
        self.labels = self.data.labels
        self.train_idx = self.data.train_idx
        # split training and validation set
        # andrà modificato se aggiungiamo il parametro test_size
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
        self.num_nodes = g.number_of_nodes()
        return g

    def fit(self, want_accuracy=True):
        print("start training...")
        self.train()

        # --------
        forward_time = []
        backward_time = []

        for epoch in range(self.n_epochs):
            # zero_grad deprecated: instead: cleargrads()
            self.optimizer.zero_grad()
            t0 = time.time()
            logits = self.RGCN.forward(self.graph)
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
        self.eval()
        # print("labels", self.labels[:5])  # 60'000
        logits = self.RGCN.forward(self.graph)  # 60'000*26 per ogni classe
        # print("logits", sum(logits[0]))
        # self.test_idx 9533 la classe predetta per ogni elemento del test
        test_loss = F.cross_entropy(logits[self.test_idx], self.labels[self.test_idx].long())
        test_acc = torch.sum(logits[self.test_idx].argmax(dim=1) == self.labels[self.test_idx]).item() / len(
            self.test_idx)
        print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
        print()

        print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
        print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))

    def _print_parameters(self):
        print(
            "\n\n******* Launching RGCN-Model Node Classifier ****************************************************************************************************************")
        # print("Dataset:", self.dataset_name)

        print("Embeddings dim:", self.h_dim)
        print("Number of hidden layers:", self.num_hidden_layers)
        print("Number of bases:", self.num_bases)
        print("Number of epochs:", self.n_epochs)
        print("Learning rate:", self.lr)
        print("L2norm:", self.l2norm)
        print()

    def predict(self):
        self.eval()
        # print("labels", self.labels[:5])  # 60'000
        logits = self.RGCN.forward(self.graph)  # 60'000*26 per ogni classe
        # print("logits", sum(logits[0]))
        # self.test_idx 9533 la classe predetta per ogni elemento del test

        # print(self.labels[self.test_idx[0]].item())
        # TODO SALVA IN UN CSV
        print("Entità   Classe-Reale    Classe-Predetta")
        for i in range(len(self.test_idx)):
            print("E" + str(self.test_idx[i]) + ":  " + str(self.labels_dict[self.labels[self.test_idx[i]].item()]) +
                  " " + str(self.labels_dict[torch.argmax(logits[self.test_idx[i]]).item()]))

        # test_loss = F.cross_entropy(logits[self.test_idx], self.labels[self.test_idx].long())
        # test_acc = torch.sum(logits[self.test_idx].argmax(dim=1) == self.labels[self.test_idx]).item() / len(
        #     self.test_idx)
        # print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
        # print()
        #
        # print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
        # print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))

        print()

    def predict_single(self, entities_vector):
        if self.check_entities_vector(entities_vector):
            self.eval()

            logits = self.RGCN.forward(self.graph)  # 60'000*26 per ogni classe

            test_idx = entities_vector
            print("Entità   Classe-Reale    Classe-Predetta")
            for i in range(len(test_idx)):
                print(
                    "E" + str(test_idx[i]) + ":  " + str(self.labels_dict[self.labels[test_idx[i]].item()]) +
                    " " + str(self.labels_dict[torch.argmax(logits[test_idx[i]]).item()]))
        else:
            print()
            print("La lista di entità da esaminare richiesta non è totalmente compresa nel datatest")
            print()

    def check_entities_vector(self, entities_vector):
        # TODO CHECK with list comprehension
        ret = True
        for i in entities_vector:
            if i not in self.test_idx:
                ret = False
                break
        return ret