import ConfigParser
import csv
import matplotlib.pyplot as plt


def write_configs(model, train_percent = 0.3, scale = 'true', normalize = 'true', k_best = 'false'):
    config = ConfigParser.RawConfigParser()

    config.add_section('TrainingData')
    config.set('TrainingData', 'labelColumn', 'label')
    config.set('TrainingData', 'textColumn', 'extracted_text')
    config.set('TrainingData', 'separateTrainingTesting', 'false')
    config.set('TrainingData', 'possibleLabels', '2,3,4')
    config.set('TrainingData', 'cross_validation', 'false')

    config.add_section('Classifier')
    config.set('Classifier', 'model', model)
    config.set('Classifier', 'scale', scale)
    config.set('Classifier', 'normalize', normalize)
    config.set('Classifier', 'k_best', k_best)
    config.set('Classifier', 'trainPercent', train_percent)

    with open('ad_classification.ini', 'wb') as configfile:
        config.write(configfile)

def read_results():
    p_r_fscore_support = dict()

    with open('results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if(index != 0):
                if(row[0] == 'Weighted'):
                    p_r_fscore_support['precision'] = float(row[1])
                    p_r_fscore_support['recall'] = float(row[2])
                    p_r_fscore_support['fscore'] = float(row[3])
            else:
                print row
    print p_r_fscore_support
    return p_r_fscore_support

def plot_graph(train_results, train_percents, model, is_multiple = True, type= 'Precision'):
    plt.clf()
    if(is_multiple):
        plt.title(type+'-Train Percentage curve')
        for index, result in enumerate(train_results):
            plt.plot(train_percents, result, lw=2 , label=model[index])
        plt.legend(loc = 'lower right')
    else:
        plt.title(model + ' '+type+'-Train Percentage curve')
        plt.plot(train_percents, train_results, lw=2, color='navy', label=model)
    plt.xlabel('Train Percent')
    plt.ylabel(type)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    if(is_multiple):
        plt.savefig(type+'all.png', dpi=100)
    else:
        plt.savefig(type+'-'+model +'.png', dpi=100)
    #plt.show()

classifiers = ['random_forest', 'knn', 'logistic_regression', 'svm', 'sgd', 'nn', 'dtree', 'gaussianNB']
#classifiers = ['random_forest','nn']
train_percents = [0.1, 0.3, 0.5, 0.7, 0.9]
#train_percents = [0.1, 0.3]
model_results_prec = [0] * len(classifiers)
model_results_rec = [0] * len(classifiers)
for i, model in enumerate(classifiers):
    train_results_prec = [0] * len(train_percents)
    train_results_rec = [0] * len(train_percents)
    for index, percent in enumerate(train_percents):
        write_configs(model, percent)
        execfile('ad_classification.py')
        stats = read_results()
        train_results_prec[index] = stats['precision']
        train_results_rec[index] = stats['recall']
    plot_graph(train_results_prec, train_percents, model, False, 'Precision')
    plot_graph(train_results_rec, train_percents, model, False, 'Recall')
    model_results_prec[i] = train_results_prec
    model_results_rec[i] = train_results_rec

plot_graph(model_results_prec, train_percents, classifiers, True, 'Precision')
plot_graph(model_results_rec, train_percents, classifiers, True, 'Recall')