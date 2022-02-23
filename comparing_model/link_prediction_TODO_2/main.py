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

tf = TriplesFactory.from_path(r'C:\Users\Girolamo\PycharmProjects\rgcnKE_sus\dataset\km4city\dataset_for_link_prediction\classification\Complete.txt')
training, testing = tf.split()
result = pipeline(
    training=training,
    testing=testing,
    model='TransE',
    epochs=2,  # short epochs for testing - you should go higher
)
result.save_to_directory('doctests/test_unstratified_transe')
