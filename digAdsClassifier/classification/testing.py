import digAdsClassifier.definitions.doc_embeddings
import digAdsClassifier.utility.functions
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, precision_recall_fscore_support
from sklearn.externals import joblib

def persist(classifier):
    if('scaler' in classifier):
        joblib.dump(classifier['scaler'], 'scaler.pkl')
        
    if('normalizer' in classifier):
        joblib.dump(classifier['normalizer'], 'normalizer.pkl')

    if('k_best' in classifier):
        joblib.dump(classifier['k_best'], 'k_best.pkl')

    joblib.dump(classifier['model'], 'model.pkl')

def test_model(feature_vectors, classifier):
    print feature_vectors.shape
    if(feature_vectors.ndim == 1):
        print "Reshaping"
        feature_vectors = feature_vectors.reshape(1, -1)

    if('scaler' in classifier):
        print "Scaling"
        feature_vectors = classifier['scaler'].transform(feature_vectors)
        
    if('normalizer' in classifier):
        print "Nomalizing"
        feature_vectors = classifier['normalizer'].transform(feature_vectors)

    if('k_best' in classifier):
        feature_vectors = classifier['k_best'].transform(feature_vectors)
    
    return classifier['model'].predict(feature_vectors)

def calculate_precision_recall(predicted_labels, test_labels):
    #for index, label in enumerate(test_labels):
    #    print "Testing:",
     #   print label,
     #   print ", Predicted:",
      #  print predicted_labels[index]

    
    with open('results.csv', 'w') as f:
        f.write('Type,Precision,Recall,F-Score,Support\n')
        prf = ['Precision: ', 'Recall: ', 'F-score: ', 'Support: ']
        f.write("Macro")
        print "Macro"
        k = precision_recall_fscore_support(test_labels, predicted_labels, average='macro')
        for i in range(0, len(k)):
            print prf[i],
            print k[i]
            f.write(",")
            f.write('%s' %k[i])
        f.write("\n")

        print "Micro"
        f.write("Micro")
        k = precision_recall_fscore_support(test_labels, predicted_labels, average='micro')
        for i in range(0, len(k)):
            print prf[i],
            print k[i]
            f.write(",")
            f.write('%s' %k[i])
        f.write("\n")

        print "Weighted"
        f.write("Weighted")
        k = precision_recall_fscore_support(test_labels, predicted_labels, average='weighted')
        for i in range(0, len(k)):
            print prf[i],
            print k[i]
            f.write(",")
            f.write('%s' %k[i])
        f.write("\n")

        print 'Class 2\tClass 3\tClass 4'
        f.write("All")
        k = precision_recall_fscore_support(test_labels, predicted_labels)
        for i in range(0, len(k)):
            print prf[i],
            print k[i]
            f.write(",")
            f.write('%s' %k[i])
        f.write("\n")
