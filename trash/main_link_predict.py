import os
import time

from trash.my_utils import *
from loader import load_link_dataset
from trash.model import *

dataset_dir = str(os.getcwd()) + "\\FB15k\\"
dataset_name = "FB15k"
# load graph data
data = load_link_dataset(dir=dataset_dir, datasets=dataset_name)

num_nodes = data.num_nodes
train_data = data.train
valid_data = data.valid
test_data = data.test
num_rels = data.num_rels

# --- PARAMETERS ---
n_hidden = 500
n_bases = 100
n_layers = 2
dropout = 0.2
regularization = 0.01
lr = 0.01
graph_batch_size = 30000
graph_split_size = 0.5
negative_sample = 10
grad_norm = 1.0
edge_sampler = "neighbor" # 'uniform' or 'neighbor'
evaluate_every = 500
eval_batch_size = 500
eval_protocol = "raw" # 'raw' or 'filterred'
n_epochs = 6000

# create model
model = LinkPredict(num_nodes,
                    n_hidden,
                    num_rels,
                    num_bases=n_bases,
                    num_hidden_layers=n_layers,
                    dropout=dropout,
                    reg_param=regularization)

# validation and testing triplets
valid_data = torch.LongTensor(valid_data)
test_data = torch.LongTensor(test_data)

# build test graph
test_graph, test_rel, test_norm = build_test_graph(
    num_nodes, num_rels, train_data)
test_deg = test_graph.in_degrees(
    range(test_graph.number_of_nodes())).float().view(-1, 1)
test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
test_rel = torch.from_numpy(test_rel)
test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))



# build adj list and calculate degrees for sampling
adj_list, degrees = get_adj_and_degrees(num_nodes, train_data)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model_state_file = '../model_state.pth'
forward_time = []
backward_time = []

# training loop
print("start training...")

epoch = 0
best_mrr = 0
while True:
    model.train()
    epoch += 1

    # perform edge neighborhood sampling to generate training graph and data
    g, node_id, edge_type, node_norm, data, labels = \
        generate_sampled_graph_and_labels(
            train_data, graph_batch_size, graph_split_size,
            num_rels, adj_list, degrees, negative_sample,
            edge_sampler)
    print("Done edge sampling")

    # set node/edge feature
    node_id = torch.from_numpy(node_id).view(-1, 1).long()
    edge_type = torch.from_numpy(edge_type)
    edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
    data, labels = torch.from_numpy(data), torch.from_numpy(labels)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)


    t0 = time.time()
    embed = model(g, node_id, edge_type, edge_norm)
    loss = model.get_loss(g, embed, data, labels)
    t1 = time.time()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)  # clip gradients
    optimizer.step()
    t2 = time.time()

    forward_time.append(t1 - t0)
    backward_time.append(t2 - t1)
    print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
          format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

    optimizer.zero_grad()

    # validation
    if epoch % evaluate_every == 0:
        # perform validation on CPU because full graph is too large

        model.eval()
        print("start eval")
        embed = model(test_graph, test_node_id, test_rel, test_norm)
        mrr = calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
                             valid_data, test_data, hits=[1, 3, 10], eval_bz=eval_batch_size,
                             eval_p=eval_protocol)
        # save best model
        if best_mrr < mrr:
            best_mrr = mrr
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

        if epoch >= n_epochs:
            break

print("training done")
print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

print("\nstart testing:")
# use best model checkpoint
checkpoint = torch.load(model_state_file)

model.eval()
model.load_state_dict(checkpoint['state_dict'])
print("Using best epoch: {}".format(checkpoint['epoch']))
embed = model(test_graph, test_node_id, test_rel, test_norm)
calc_mrr(embed, model.w_relation, torch.LongTensor(train_data), valid_data,
               test_data, hits=[1, 3, 10], eval_bz=eval_batch_size, eval_p=eval_protocol)
