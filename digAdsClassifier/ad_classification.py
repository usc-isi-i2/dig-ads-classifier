import os
import sys
import utility.functions
import preprocess.embeddings
import classification.training
import classification.testing
import classification.cross_validate
import sys
reload(sys)
sys.setdefaultencoding('utf8')

UNIGRAM_FILE = 'unigram-part-00000-v2.json'
TRAINING_FILE = 'training.csv'
TESTING_FILE = 'testing.csv'
CONFIG_FILE = 'ad_classification.ini'
RESOURCES_FOLDER = 'resources' #Should be present in the path of this file

#Preprocess the documents using the embeddings file

def create_doc_embeddings(data_file, embeddings, config):
    """
    Creates Doc Embeggings for the data file using the embeddings file
    """
    utility.functions.print_name()
    docs = preprocess.embeddings.get_doc_embeddings(data_file, embeddings, config)
    #print docs.data
    return docs

def classify_docs(config, doc_feature_vectors_training, doc_feature_vectors_testing = None):
    """
    Classifies the feature vectors passed
    """
    utility.functions.print_name()
    doc_feature_vectors = classification.training.mark_train_test_data(config, doc_feature_vectors_training, doc_feature_vectors_testing)
    #print doc_feature_vectors.data
    training_testing_data = classification.training.extract_training_testing_data(doc_feature_vectors)
    classifier = classification.training.train_model(config, training_testing_data['feature_vectors_train'], training_testing_data['labels_train'])

    predicted_labels = classification.testing.test_model(training_testing_data['feature_vectors_test'], classifier)
    classification.testing.calculate_precision_recall(predicted_labels, training_testing_data['labels_test'])

    classification.testing.persist(classifier)

def cross_validate(config, doc_feature_vectors_training):
    """
    Cross Validate the given feature vectors
    """
    utility.functions.print_name()
    doc_feature_vectors = classification.training.mark_train_test_data(config, doc_feature_vectors_training)
    training_testing_data = classification.training.extract_training_testing_data(doc_feature_vectors)

    model_parameters = classification.cross_validate.get_model_parameters(config)
    grid_search = classification.cross_validate.get_grid_search(config, model_parameters)
    
    classifier = classification.cross_validate.run_grid_search(config, grid_search, training_testing_data['feature_vectors_train'], training_testing_data['labels_train'])
    classification.cross_validate.build_statistics(grid_search)

    #Testing on the best model
    predicted_labels = classification.testing.test_model(training_testing_data['feature_vectors_test'], classifier)
    classification.testing.calculate_precision_recall(predicted_labels, training_testing_data['labels_test'])

    #Training and testing on the default params model as well
    classifier = classification.training.train_model(config, training_testing_data['feature_vectors_train'], training_testing_data['labels_train'])
    predicted_labels = classification.testing.test_model(training_testing_data['feature_vectors_test'], classifier)
    classification.testing.calculate_precision_recall(predicted_labels, training_testing_data['labels_test'])


current_path = os.path.dirname(os.path.abspath(__file__))
resource = utility.functions.ResourcesFile(current_path, RESOURCES_FOLDER)
config = utility.functions.AdConfig(CONFIG_FILE)
embeddings = preprocess.embeddings.create(resource.get(UNIGRAM_FILE))
doc_embeddings_training = create_doc_embeddings(resource.get(TRAINING_FILE), embeddings, config)

if(config.get("cross_validation")):
    #Need to cross validate instead of testing training data
    #Use the training data to do k fold cross validation
    cross_validate(config, doc_embeddings_training)

else:
    #Will be training and testing on separate parts
    if(config.get("separateTrainingTesting")):
        #There are separate training testing files
        doc_embeddings_testing = create_doc_embeddings(resource.get(TESTING_FILE), embeddings, config)
        classify_docs(config, doc_embeddings_training, doc_embeddings_testing)
    else:
        #Use some part of the file as training data and rest as training, depening on configuration
        classify_docs(config, doc_embeddings_training)