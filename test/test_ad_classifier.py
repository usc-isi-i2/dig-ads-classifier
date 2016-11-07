import os
import sys
import codecs

import unittest

import json
from digExtractor.extractor_processor import ExtractorProcessor
import digAdsClassifier.ads_classifier
import digAdsClassifier.utility.functions
import digAdsClassifier.preprocess.embeddings

UNIGRAM_FILE = 'unigram-part-00000-v2.json'

class TestAdClassifier(unittest.TestCase):

    def load_embeddings(self, filename):
        names_file = os.path.join(os.path.dirname(__file__), filename)
        print names_file
        return digAdsClassifier.preprocess.embeddings.create(names_file)

    def test_ad_classifier(self):
        embeddings = self.load_embeddings(UNIGRAM_FILE)
        doc = {"readability_text": "Massage in London SW1 | victoriatantric.co.uk | Tantric I added my business 'Voluptas Tantric Massage' to CityLocal, the premier business directory in Westminster If you'...  http://t.co/QYKzbNYgOY Voluptas Tantric Massage in Victoria is a great place to spend and hour or two. Luxury Apartment for that special London massage. Tantric massage London SW1"}
        extractor = digAdsClassifier.ads_classifier.AdsClassifier()
        extractor.set_embeddings(embeddings)
        extractor_processor = ExtractorProcessor().set_input_fields('readability_text').set_output_field('ad_type').set_extractor(extractor)

        updated_doc = extractor_processor.extract(doc)

        self.assertEquals(updated_doc['ad_type'][0]['value'], 'massage-parlor')

if __name__ == '__main__':
    unittest.main()
