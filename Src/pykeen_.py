# from pykeen.triples import TriplesFactory
# from pykeen.pipeline import pipeline
# NATIONS_TRAIN_PATH = "C:\\Users\\mistr\\OneDrive\\Desktop\\pythonProject\\rgcnKE\\dataset\\km4city\\dataset_for_link_prediction\\classification\\train.txt"
# NATIONS_TEST_PATH = "C:\\Users\\mistr\\OneDrive\\Desktop\\pythonProject\\rgcnKE\\dataset\\km4city\\dataset_for_link_prediction\\classification\\test.txt"
#
# result = pipeline(
#     training=NATIONS_TRAIN_PATH,
#     testing=NATIONS_TEST_PATH,
#     model='R-GCN',
#     epochs=2)
# result.save_to_directory('doctests/test_pre_stratified_transe')

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from Src.DataManagement.reformat_data import reformat_data
import configparser
import os

class pykeen_link_prediction():

    def __init__(self, directory, dataset='km4city'):
        self.dataset = dataset
        self.directory = directory
        tf = TriplesFactory.from_path(r"{}/{}/LinkPrediction/PyKeen/complete.txt".format(self.directory, self.dataset))
        self.training, self.testing = tf.split()
        config = configparser.ConfigParser()
        config.read(r'{}\parameters\pykeen.ini'.format(self.directory.replace('dataset', "Src")))
        self.model = config.get('config', 'model')
        self.epochs = config.getint('config', 'epochs')

    def fit(self):
        self.result = pipeline(
            training=self.training,
            testing=self.testing,
            model=self.model,
            epochs=self.epochs,
        )
        self.result.save_to_directory('doctests/test_unstratified_transe') #todo capire come fare

if __name__=='__main__':
    reformat_data(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset","km4city",data_type="pykeen")
    pykeen_link_prediction(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset").fit()