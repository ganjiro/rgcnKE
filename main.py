
# num_nodes: numero di entità
# emb_dim: emb
# out_dim:
# num_rels: numero di tipi di relazioni
# num_bases:

#
# class BaseRGCN(nn.Module):
#     def __init__(self, num_nodes, emb_dim, out_dim, num_rels, num_bases,
#                  num_hidden_layers=1,
#                  use_self_loop=False):
#         super(BaseRGCN, self).__init__()
#         self.num_nodes = num_nodes
#         self.emb_dim = emb_dim
#         self.out_dim = out_dim
#         self.num_rels = num_rels
#         self.num_bases = None if num_bases < 0 else num_bases # ???
#         self.num_hidden_layers = num_hidden_layers # ???
#         self.use_self_loop = use_self_loop # va scelto nel caso in cui si faccia link o node classification
#         # create rgcn layers
#         self.build_model()
#
#     def build_model(self):
#         self.layers = nn.ModuleList()
#         # i2h
#         i2h = self.build_input_layer()
#         if i2h is not None:
#             self.layers.append(i2h)
#         # h2h
#         for idx in range(self.num_hidden_layers):
#             h2h = self.build_hidden_layer(idx)
#             self.layers.append(h2h)
#         # h2o
#         h2o = self.build_output_layer()
#         if h2o is not None:
#             self.layers.append(h2o)
#
#     def build_input_layer(self):
#         return None
#
#     def build_hidden_layer(self, idx):
#         raise NotImplementedError
#
#     def build_output_layer(self):
#         return None
#
#     def forward(self, g, h, r, norm):
#         for layer in self.layers:
#             h = layer(g, h, r, norm)
#         return h
#
#
# def initializer(emb):
#     emb.uniform_(-1.0, 1.0)
#     return emb
#
