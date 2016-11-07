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
from sklearn.model_selection import GridSearchCV 

def build_statistics(grid_search):
    print("Best parameters set found on development set:")
    print()
    print(grid_search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

def run_grid_search(config, grid_search, feature_vectors, labels, classifier_model = 'svm', scale = True, normalize = False, kBest = False):
    
    if(config.has("model")):
        classifier_model = config.get("model")
        scale = config.get("scale")
        normalize = config.get("normalize")

    print classifier_model
    print 'Scale:',
    print scale
    print 'Normalize:',
    print normalize

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

    if classifier_model == 'svm':
        grid_search.fit(feature_vectors, labels)

    classifier['model'] = grid_search
    return classifier

def get_model_parameters(config, classifier_model = 'svm'):
    if(config.has("model")):
        classifier_model = config.get("model")

    if(classifier_model == 'svm'):
        return [
        {
        'C': [1e-2, 1e-1, 1, 1e+2, 1e+4, 1e+8],
        'loss': ['squared_hinge'],
        'penalty': ['l1','l2'],
        'dual':[False],
        'tol': [1e-8, 1e-6, 1e-4, 1e-2, 1],
        'multi_class': ['ovr', 'crammer_singer'],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']
        },
        {
        'C': [1e-2, 1e-1, 1, 1e+2, 1e+4, 1e+8],
        'loss': ['hinge'],
        'penalty': ['l2'],
        'dual':[True],
        'tol': [1e-8, 1e-6, 1e-4, 1e-2, 1],
        'multi_class': ['ovr', 'crammer_singer'],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']
        },
        {
        'C': [1e-2, 1e-1, 1, 1e+2, 1e+4, 1e+8],
        'loss': ['squared_hinge'],
        'penalty': ['l2'],
        'dual':[True],
        'tol': [1e-8, 1e-6, 1e-4, 1e-2, 1],
        'multi_class': ['ovr', 'crammer_singer'],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']
        }
        ]
    else:
        raise Exception("The classifier not supported")

def get_grid_search(config, model_parameters, classifier_model = 'svm'):
    if(config.has("model")):
        classifier_model = config.get("model")

    model = _get_model(classifier_model)
    scoring = _get_scoring()

    return GridSearchCV(model, model_parameters, cv=5, scoring=scoring)

def _get_model(classifier_model):
    if(classifier_model == 'svm'):
        return svm.LinearSVC()

def _get_scoring():
    return 'f1_weighted'