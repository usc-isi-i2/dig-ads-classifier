from sklearn.externals import joblib
import digAdsClassifier.utility.functions
import digAdsClassifier.preprocess.embeddings
import digAdsClassifier.classification.testing


class ClassifyAds:
    def __init__(self, model_filenames, embeddings):
        self.embeddings = embeddings
        self.classifier = dict()
        try:
            self.classifier['model'] = joblib.load(model_filenames[0])
        except:
            raise Exception('Model file not present')
        try:
            self.classifier['scaler'] = joblib.load(model_filenames[1])
            self.classifier['normalizer'] = joblib.load(model_filenames[2])
            self.classifier['k_best'] = joblib.load(model_filenames[3])
        except:
            pass

    def map_label(self, predicted_label):
        if(predicted_label == 2):
            return "massage-parlor"
        if(predicted_label == 3):
            return "escort"
        if(predicted_label == 4):
            return "job-ad"

    def run(self, text):
        print text
        word_tokens = digAdsClassifier.utility.functions.get_word_tokens(text)
        embedding = digAdsClassifier.preprocess.embeddings.combine_word_embeddings(word_tokens, self.embeddings)
        predicted_label = digAdsClassifier.classification.testing.test_model(embedding, self.classifier)
        
        return self.map_label(predicted_label[0])
