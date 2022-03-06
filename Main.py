import os
from Src.DataManagement.reformat_data import reformat_data
from Src.DataManagement.download_raw_dataset import download_km4c
from Src.minwalc import MINDWALC_node_classification
from Src.noge import NoGE_link_prediction
from Src.rgcn import RGCN_node_classification
from Src.pykeen_ import pykeen_link_prediction

if __name__=='__main__':
    path_to_dataset = r'{}\dataset'.format(os.getcwd())
    #download_km4c(file_path=r"{}/km4c/RawDowloaded/subgraph_data.TSV".format(path_to_dataset), want_tsv=True)

    #reformat_data(path_to_dataset,"km4city",data_type="pykeen")

  #  MINDWALC_node_classification(path_to_dataset).fit()

   # NoGE_link_prediction(path_to_dataset).fit()

   # RGCN_node_classification(path_to_dataset).fit()

    pykeen_link_prediction(path_to_dataset).fit()