import copy

from digExtractor.extractor import Extractor
import digAdsClassifier.classify_ads

class AdsClassifier(Extractor):

    def __init__(self):
        self.renamed_input_fields = 'readability_text'
        self.metadata = {"extractor": "AdsClassifier"}
        self.model_filenames = ['model.pkl','scaler.pkl', 'normalizer.pkl', 'k_best.pkl']
        self.embeddings = {}
        #joblib.dump(clf, 'filename.pkl')

    def extract(self, doc):
        #try:
            if 'readability_text' in doc:
                classifier = digAdsClassifier.classify_ads.ClassifyAds(self.model_filenames, self.embeddings)
                return classifier.run(doc['readability_text'])
            else:
                return None
        #except:
        #    return "Failed"

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_embeddings():
        return self.embeddings

    def get_metadata(self):
        """Returns a copy of the metadata that characterizes this extractor"""
        return copy.copy(self.metadata)

    def set_metadata(self, metadata):
        """Overwrite the metadata that characterizes this extractor"""
        self.metadata = metadata
        return self

    def get_renamed_input_fields(self):
        """Return a scalar or ordered list of fields to rename to"""
        return self.renamed_input_fields

    def get_model_filenames(self):
        return self.model_filenames

    def set_model_filenames(self, filenames):
        self.model_filenames = filenames
        return self