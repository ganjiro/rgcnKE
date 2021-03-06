import os
import time
from src.DataManagement.reformat_data import reformat_data
from src.DataManagement.download_raw_dataset import download_km4c
from src.mindwalc import MINDWALC_node_classification
from src.noge import NoGE_link_prediction
from src.rgcn import RGCN_node_classification
from src.testpykeen import pykeen_link_prediction

if __name__ == '__main__':
    times = []
    path_to_dataset = r'{}\dataset'.format(os.getcwd())

    download_km4c(file_path=r"{}/km4city/RawDowloaded/subgraph_data.TSV".format(path_to_dataset), want_tsv=True)

    reformat_data(path_to_dataset, "km4city", data_type="node")  # per node classification
    reformat_data(path_to_dataset, "km4city", data_type="noge")
    reformat_data(path_to_dataset, "km4city", data_type="pykeen")

    start = time.time()
    rgcn = RGCN_node_classification(path_to_dataset)
    rgcn.fit()
    rgcn.predict()
    end = time.time()
    times.append(end-start)

    start = time.time()
    dq_gnn = NoGE_link_prediction(path_to_dataset)
    dq_gnn.fit()
    end = time.time()
    times.append(end - start)

    start = time.time()
    decision_tree = MINDWALC_node_classification(path_to_dataset)
    decision_tree.fit()
    decision_tree.predict()
    end = time.time()
    times.append(end - start)

    start = time.time()
    tuckER = pykeen_link_prediction(path_to_dataset)
    tuckER.fit()
    end = time.time()
    times.append(end - start)

    print("times: ", times)