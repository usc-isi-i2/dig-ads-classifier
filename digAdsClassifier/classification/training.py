import definitions.doc_embeddings
import utility.functions
import math
import random
from numpy import vstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import neighbors
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import chi2, f_classif, SelectKBest

def train_model(config, feature_vectors, labels, classifier_model = 'random_forest', scale = True, normalize = False, kBest = False):

    if(config.has("model")):
        classifier_model = config.get("model")
        scale = config.get("scale")
        normalize = config.get("normalize")
        kBest = config.get("k_best")

    print classifier_model
    print 'Scale:',
    print scale
    print 'Normalize:',
    print normalize
    print 'K-Best',
    print kBest

    classifier = dict()

    if(scale):
        scaler = StandardScaler()
        scaler.fit(feature_vectors)
        feature_vectors = scaler.transform(feature_vectors)
        classifier['scaler'] = scaler

    if(normalize):
        normalizer = Normalizer()
        normalizer.fit(feature_vectors)
        feature_vectors = normalizer.transform(feature_vectors)
        classifier['normalizer'] = normalizer

    if(kBest):
        kBest = SelectKBest(f_classif, k=20)
        kBest = kBest.fit(feature_vectors, labels)
        feature_vectors = kBest.transform(feature_vectors)
        classifier['k_best'] = kBest

    #print feature_vectors.shape
    if classifier_model == 'random_forest':
        model = RandomForestClassifier()
        model.fit(feature_vectors, labels)
    elif classifier_model == 'knn':
        k = 3
        model = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
        model.fit(feature_vectors, labels)
    elif classifier_model == 'logistic_regression':
        model = LogisticRegression()
        model.fit(feature_vectors, labels)
    elif classifier_model == 'svm':
        model = svm.LinearSVC()
        model.fit(feature_vectors, labels)
    elif classifier_model == 'sgd':
        model = SGDClassifier(loss="modified_huber", penalty="l1")
        model.fit(feature_vectors, labels)
    elif classifier_model == 'nn':
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        model.fit(feature_vectors, labels)
    elif classifier_model == 'dtree':
        model = tree.DecisionTreeClassifier()
        model.fit(feature_vectors, labels)
    elif classifier_model == 'gaussianNB':
        model = GaussianNB()
        model.fit(feature_vectors, labels)

    classifier['model'] = model
    return classifier


def extract_training_testing_data(doc_feature_vectors):
    feature_vectors_train = None
    labels_train = []
    feature_vectors_test = None
    labels_test = []
    #print len(doc_feature_vectors.data)
    for index, doc in enumerate(doc_feature_vectors.data):
        if(doc_feature_vectors.is_train(index)):
            #print "Is train"
            feature_vectors_train = _add_vector_and_label(feature_vectors_train, labels_train, doc_feature_vectors, index)
        else:
            #print "Is test"
            if(doc_feature_vectors.data[index]['embedding'] is None):
                print "Is None"
            else:
                feature_vectors_test = _add_vector_and_label(feature_vectors_test, labels_test, doc_feature_vectors, index)
    training_testing_data = {}
    training_testing_data['feature_vectors_train'] = feature_vectors_train
    training_testing_data['labels_train'] = labels_train
    training_testing_data['feature_vectors_test'] = feature_vectors_test
    training_testing_data['labels_test'] = labels_test
    
    return training_testing_data

def _add_vector_and_label(feature_vectors, labels, doc_feature_vectors, index):
    labels.append(doc_feature_vectors.data[index]['label'])
    if(feature_vectors is not None):
        #print feature_vectors.shape
        #print doc_feature_vectors.data[index]['embedding'].shape
        return vstack((feature_vectors, doc_feature_vectors.data[index]['embedding']))
    else:
        return doc_feature_vectors.data[index]['embedding']
    

def mark_train_test_data(config, doc_feature_vectors_training, doc_feature_vectors_testing = None):
    """
    Marks the doc feature vectors as either testing or training data

    """
    if(config.get("separateTrainingTesting")):
        #There is a separate training testing file. Putting all data from 1st to training the model. Put all data from 2nd to testing the model
        _mark_train_test_data(config, doc_feature_vectors_training, train_percent = 1.0, randomized = False)
        if(doc_feature_vectors_testing is not None):
            _mark_train_test_data(config, doc_feature_vectors_testing, train_percent = 0.0, randomized = False, balanced = False)
            doc_feature_vectors_training.append(doc_feature_vectors_testing)
        return doc_feature_vectors_training

    else:
        #Take a specific percentage of data as training, rest as testing
        _mark_train_test_data(config, doc_feature_vectors_training)
        return doc_feature_vectors_training

def _mark_train_test_data(config, doc_feature_vectors, train_percent=0.3, randomized = True, balanced = True):

    if(not config.get("separateTrainingTesting") and config.has("trainPercent")):
        train_percent = config.get("trainPercent")

    print 'Train Percent:',
    print train_percent

    number_of_points = len(doc_feature_vectors.data)
    number_of_points_training = int(math.ceil(train_percent * number_of_points))
    number_of_points_testing = number_of_points - number_of_points_training

    print number_of_points
    print number_of_points_training
    print number_of_points_testing

    if(randomized):
        training_indices = random.sample(xrange(number_of_points), number_of_points_training)

    else:
        training_indices = range(0,number_of_points_training)

    #print training_indices
    _mark_indices_as_train_test(training_indices, doc_feature_vectors)

    if(balanced):

        possible_labels = _get_possible_labels(config)

        count_of_labels = _get_count_of_labels(possible_labels, doc_feature_vectors)

        max_count = max(count_of_labels)
        min_count = min(count_of_labels)

        if(min_count == 0):
            #There is no sample selected of a particular type
            _exchange_sample_if_present(doc_feature_vectors, possible_labels, count_of_labels, training_indices)

        #Now there is atleast one sample of each possible label
        _balance_sampling(doc_feature_vectors, possible_labels, count_of_labels, max_count, training_indices)

        _get_count_of_labels(possible_labels, doc_feature_vectors)       

def _mark_indices_as_train_test(training_indices, doc_feature_vectors):
    for i in range(0,len(doc_feature_vectors.data)):
        if i in training_indices:
            doc_feature_vectors.mark_as_train(i)
        else:
            if(doc_feature_vectors.data[i]['embedding'] is None):
                print "Is None:"
                print doc_feature_vectors.data[i]
            doc_feature_vectors.mark_as_test(i)


def _get_count_of_labels(possible_labels, doc_feature_vectors):
    count_of_labels = [0] * len(possible_labels)
    for i in range(0, len(doc_feature_vectors.data)):
        if(doc_feature_vectors.is_train(i)):
            index = possible_labels.index(int(doc_feature_vectors.data[i]['label']))
            count_of_labels[index] += 1
    print count_of_labels
    return count_of_labels

def _get_possible_labels(config):
    possible_labels = config.get("possibleLabels")
    possible_labels = possible_labels.split(",")
    possible_labels = map(int, possible_labels)
    print possible_labels
    return possible_labels

def _exchange_sample_if_present(doc_feature_vectors, possible_labels, count_of_labels, training_indices):
    """
    If there a particula label is not present in sample, try to select that label into the sample, if present
    """
    #TODO: Implement the function
    raise Exception("There is no sample of a label in the training data")

def _balance_sampling(doc_feature_vectors, possible_labels, count_of_labels, max_count, training_indices):
    for index, label in enumerate(possible_labels):
        if(count_of_labels[index] < max_count):
            count_to_resample = (max_count - count_of_labels[index])
            _resample_of_label(label, count_to_resample, doc_feature_vectors, training_indices)

def _resample_of_label(label, count_to_resample, doc_feature_vectors, training_indices):
    indices_of_this_sample = []
    for training_index in training_indices:
        if(int(doc_feature_vectors.data[training_index]['label']) == label):
            indices_of_this_sample.append(training_index)
    number_of_samples = len(indices_of_this_sample)
    print count_to_resample,
    print " from ",
    print number_of_samples
    if(count_to_resample > number_of_samples):
        repeat_count = int(math.ceil(count_to_resample/number_of_samples)) + 1
        indices_of_this_sample = indices_of_this_sample * repeat_count

    resample_indices = random.sample(xrange(len(indices_of_this_sample)), count_to_resample)

    for resample_index in resample_indices:
        doc_feature_vectors.data.append(doc_feature_vectors.data[indices_of_this_sample[resample_index]])