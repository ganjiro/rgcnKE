import configparser

from pykeen.losses import BCEWithLogitsLoss
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


class pykeen_link_prediction():

    def __init__(self, directory, dataset='km4city'):
        self.dataset = dataset
        self.directory = directory
        tf = TriplesFactory.from_path(r"{}/{}/LinkPrediction/PyKeen/complete.txt".format(self.directory, self.dataset))
        self.training, self.testing = tf.split()
        config = configparser.ConfigParser()
        config.read(r'{}\parameters\pykeen.ini'.format(self.directory.replace('dataset', "src")))
        self.model = config.get('config', 'model')
        self.epochs = config.getint('config', 'epochs')
        self.embedding_dim = config.getint('config', 'embedding_dim')

    def fit(self):
        print("************** Launching tuckER fit... ************************")
        self.result = pipeline(
            training=self.training,
            testing=self.testing,
            model=self.model,
            epochs=self.epochs,
            loss=BCEWithLogitsLoss,
            model_kwargs=dict(
                embedding_dim=self.embedding_dim,
            )
        )
        self.result.save_to_directory('doctests/test_unstratified_transe')
        print("************** ending ************************\n\n\n")


if __name__ == '__main__':
    #     reformat_data(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset", "km4city", data_type="pykeen")
    pykeen_link_prediction(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset").fit()
