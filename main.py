import os

from src.rgcn import RGCN_node_classification

if __name__ == '__main__':
    path_to_dataset = r'{}\dataset'.format(os.getcwd())

    # download_km4c(file_path=r"{}/km4city/RawDowloaded/subgraph_data.TSV".format(path_to_dataset), want_tsv=True)

    # reformat_data(path_to_dataset, "km4city", data_type="node")  # per node classification
    # reformat_data(path_to_dataset, "km4city", data_type="noge")
    # reformat_data(path_to_dataset, "km4city", data_type="pykeen")

    rgcn = RGCN_node_classification(path_to_dataset)
    rgcn.fit()
    rgcn.predict()

    # dq_gnn = NoGE_link_prediction(path_to_dataset)
    # dq_gnn.fit()

    # decision_tree = MINDWALC_node_classification(path_to_dataset)
    # decision_tree.fit()
    # decision_tree.predict()

    # tuckER = pykeen_link_prediction(path_to_dataset)
    # tuckER.fit()
