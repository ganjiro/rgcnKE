import configparser

from src.NoGE.load_data import Data
from src.NoGE.main_NoGE import NoGE


class NoGE_link_prediction():

    def __init__(self, directory, dataset='km4city'):
        self.dataset = dataset
        self.directory = directory
        config = configparser.ConfigParser()
        config.read(r'{}\parameters\noge.ini'.format(self.directory.replace('dataset', "src")))
        self.random_split = config.getboolean('config', 'random_split')
        self.encoder = config.get('config', 'encoder')
        self.decoder = config.get('config', 'decoder')
        self.num_iterations = config.getint('config', 'num_iterations')
        self.batch_size = config.getint('config', 'batch_size')
        self.learning_rate = config.getfloat('config', 'learning_rate')
        self.label_smoothing = config.getfloat('config', 'label_smoothing')
        self.hidden_dim = config.getint('config', 'hidden_dim')
        self.emb_dim = config.getint('config', 'emb_dim')
        self.num_layers = config.getint('config', 'num_layers')
        self.variant = config.get('config', 'variant')
        self.eval_step = config.getint('config', 'eval_step')
        self.eval_after = config.getint('config', 'eval_after')

        d = Data(data_dir=r"{}/{}/LinkPrediction/NoGE/".format(self.directory, self.dataset))

        self.model = NoGE(data=d, encoder=self.encoder, decoder=self.decoder, num_iterations=self.num_iterations,
                          batch_size=self.batch_size,
                          learning_rate=self.learning_rate, hidden_dim=self.hidden_dim, emb_dim=self.emb_dim,
                          num_layers=self.num_layers,
                          eval_step=self.eval_step, eval_after=self.eval_after, variant=self.variant)

    def fit(self):
        print("************** Launching DQ-GNN fit... ************************")
        self.model.train_and_eval()
        print("************** ending ************************\n\n\n")



if __name__ == '__main__':
    #reformat_data(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset", "km4city", data_type="Noge")
    NoGE_link_prediction(r"C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset").fit()
