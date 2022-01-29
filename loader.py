import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dgl.contrib.data.knowledge_graph import _save_sparse_csr, RDFReader, _load_sparse_csr, _bfs_relational


class LinkDatasetLoader(object):

    def __init__(self, name, directory):
        self.name = name
        self.dir = directory

    def load(self):
        entity_path = os.path.join(self.dir, 'entities.dict')
        relation_path = os.path.join(self.dir, 'relations.dict')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = np.asarray(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        self.valid = np.asarray(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        self.test = np.asarray(_read_triplets_as_list(test_path, entity_dict, relation_dict))
        self.num_nodes = len(entity_dict)
        print("# entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        print("# relations: {}".format(self.num_rels))
        print("# edges: {}".format(len(self.train)))


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d


def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l


class DatasetLoader(object):

    def __init__(self, name, dir, label_header, nodes_header):
        self.name = name
        self.dir = dir
        self.label_header = label_header
        self.nodes_header = nodes_header

    def load(self, bfs_level=2, relabel=False):
        self.num_nodes, edges, self.num_rels, self.labels, labeled_nodes_idx, self.train_idx, self.test_idx = self._load_data(
            self.name, self.label_header, self.nodes_header, self.dir)

        # bfs to reduce edges
        if bfs_level > 0:
            print("removing nodes that are more than {} hops away".format(bfs_level))
            print()
            row, col, edge_type = edges.transpose()
            A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(self.num_nodes, self.num_nodes))
            bfs_generator = _bfs_relational(A, labeled_nodes_idx)
            lvls = list()
            lvls.append(set(labeled_nodes_idx))
            for _ in range(bfs_level):
                lvls.append(next(bfs_generator))
            to_delete = list(set(range(self.num_nodes)) - set.union(*lvls))
            eid_to_delete = np.isin(row, to_delete) + np.isin(col, to_delete)
            eid_to_keep = np.logical_not(eid_to_delete)
            self.edge_src = row[eid_to_keep]
            self.edge_dst = col[eid_to_keep]
            self.edge_type = edge_type[eid_to_keep]

            if relabel:
                uniq_nodes, edges = np.unique((self.edge_src, self.edge_dst), return_inverse=True)
                self.edge_src, self.edge_dst = np.reshape(edges, (2, -1))
                node_map = np.zeros(self.num_nodes, dtype=int)
                self.num_nodes = len(uniq_nodes)
                node_map[uniq_nodes] = np.arange(self.num_nodes)
                self.labels = self.labels[uniq_nodes]
                self.train_idx = node_map[self.train_idx]
                self.test_idx = node_map[self.test_idx]
                print("{} nodes left".format(self.num_nodes))
        else:
            self.edge_src, self.edge_dst, self.edge_type = edges.transpose()

        # normalize by dst degree
        _, inverse_index, count = np.unique((self.edge_dst, self.edge_type), axis=1, return_inverse=True,
                                            return_counts=True)
        degrees = count[inverse_index]
        self.edge_norm = np.ones(len(self.edge_dst), dtype=np.float32) / degrees.astype(np.float32)

        # convert to pytorch label format
        self.num_classes = self.labels.shape[1]
        self.labels = np.argmax(self.labels, axis=1)

    def _load_data(self, dataset_str, label_header, nodes_header, dataset_path=None):

        print('******* Loading dataset', dataset_str, "***********************")

        graph_file = os.path.join(dataset_path, '{}_stripped.nt.gz'.format(dataset_str))
        task_file = os.path.join(dataset_path, 'completeDataset.tsv')
        train_file = os.path.join(dataset_path, 'trainingSet.tsv')
        test_file = os.path.join(dataset_path, 'testSet.tsv')

        edge_file = os.path.join(dataset_path, 'edges.npz')
        labels_file = os.path.join(dataset_path, 'labels.npz')
        train_idx_file = os.path.join(dataset_path, 'train_idx.npy')
        test_idx_file = os.path.join(dataset_path, 'test_idx.npy')

        if os.path.isfile(edge_file) and os.path.isfile(labels_file) and \
                os.path.isfile(train_idx_file) and os.path.isfile(test_idx_file):

            # load precomputed adjacency matrix and labels
            all_edges = np.load(edge_file)
            num_node = all_edges['n'].item()
            edge_list = all_edges['edges']
            num_rel = all_edges['nrel'].item()

            print('Number of nodes: ', num_node)
            print('Number of edges: ', len(edge_list))
            print('Number of relations: ', num_rel)

            labels = _load_sparse_csr(labels_file)
            labeled_nodes_idx = list(labels.nonzero()[0])

            print('Number of classes: ', labels.shape[1])

            train_idx = np.load(train_idx_file)
            test_idx = np.load(test_idx_file)

        else:

            # loading labels of nodes
            labels_df = pd.read_csv(task_file, sep='\t', encoding='utf-8')
            labels_train_df = pd.read_csv(train_file, sep='\t', encoding='utf8')
            labels_test_df = pd.read_csv(test_file, sep='\t', encoding='utf8')

            with RDFReader(graph_file) as reader:

                relations = reader.relationList()
                subjects = reader.subjectSet()
                objects = reader.objectSet()

                nodes = list(subjects.union(objects))
                num_node = len(nodes)
                num_rel = len(relations)
                num_rel = 2 * num_rel + 1  # +1 is for self-relation

                assert num_node < np.iinfo(np.int32).max
                print('Number of nodes: ', num_node)
                print('Number of relations: ', num_rel)

                relations_dict = {rel: i for i, rel in enumerate(list(relations))}
                nodes_dict = {node: i for i, node in enumerate(nodes)}

                edge_list = []
                # self relation
                for i in range(num_node):
                    edge_list.append((i, i, 0))

                for i, (s, p, o) in enumerate(reader.triples()):
                    src = nodes_dict[s]
                    dst = nodes_dict[o]
                    assert src < num_node and dst < num_node
                    rel = relations_dict[p]
                    # relation id 0 is self-relation, so others should start with 1
                    edge_list.append((src, dst, 2 * rel + 1))
                    # reverse relation
                    edge_list.append((dst, src, 2 * rel + 2))

                # sort indices by destination
                edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
                edge_list = np.asarray(edge_list, dtype=int)
                print('Number of edges: ', len(edge_list))

                np.savez(edge_file, edges=edge_list, n=np.asarray(num_node), nrel=np.asarray(num_rel))

            nodes_u_dict = {np.compat.unicode(self.to_unicode(key)): val for key, val in
                            nodes_dict.items()}

            labels_set = set(labels_df[label_header].values.tolist())
            labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}

            print('{} classes: {}'.format(len(labels_set), labels_set))

            labels = sp.lil_matrix((num_node, len(labels_set)))
            labeled_nodes_idx = []

            print('Loading training set')

            train_idx = []
            train_names = []
            for nod, lab in zip(labels_train_df[nodes_header].values,
                                labels_train_df[label_header].values):
                nod = np.compat.unicode(self.to_unicode(nod))
                if nod in nodes_u_dict:
                    labeled_nodes_idx.append(nodes_u_dict[nod])
                    label_idx = labels_dict[lab]
                    labels[labeled_nodes_idx[-1], label_idx] = 1
                    train_idx.append(nodes_u_dict[nod])
                    train_names.append(nod)
                else:
                    print(u'Node not in dictionary, skipped: ',
                          nod.encode('utf-8', errors='replace'))

            print('Loading test set')

            test_idx = []
            test_names = []
            for nod, lab in zip(labels_test_df[nodes_header].values,
                                labels_test_df[label_header].values):
                nod = np.compat.unicode(self.to_unicode(nod))
                if nod in nodes_u_dict:
                    labeled_nodes_idx.append(nodes_u_dict[nod])
                    label_idx = labels_dict[lab]
                    labels[labeled_nodes_idx[-1], label_idx] = 1
                    test_idx.append(nodes_u_dict[nod])
                    test_names.append(nod)
                else:
                    print(u'Node not in dictionary, skipped: ',
                          nod.encode('utf-8', errors='replace'))

            labeled_nodes_idx = sorted(labeled_nodes_idx)
            labels = labels.tocsr()
            print('Number of classes: ', labels.shape[1])

            _save_sparse_csr(labels_file, labels)

            np.save(train_idx_file, train_idx)
            np.save(test_idx_file, test_idx)

        return num_node, edge_list, num_rel, labels, labeled_nodes_idx, train_idx, test_idx

    def to_unicode(self, input):
        return str(input)


def load_dataset(label_header, nodes_header, datasets, dir, bfs_level=3):
    data = DatasetLoader(datasets, dir, label_header, nodes_header)
    data.load(bfs_level, False)
    return data


def load_link_dataset(datasets, dir):
    data = LinkDatasetLoader(datasets, dir)
    data.load()
    return data


if __name__ == '__main__':
    label_header = 'label'
    nodes_header = 'nodes'
    datasets = 'km4c'
    curr_dir = str(os.getcwd()) + '\\km4city\\'
    n_hop = 3  # di default pari a 2

    data = DatasetLoader(datasets, curr_dir, label_header, nodes_header)
    data.load(n_hop)
