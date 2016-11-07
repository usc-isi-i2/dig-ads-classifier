import codecs
import json

class Embeddings:
    def __init__(self, embeddings_file):
        self.dict = dict()
        with codecs.open(embeddings_file, 'r', 'utf-8') as f:
            for line in f:
                obj = json.loads(line)
                for k, v in obj.items():
                    self.dict[k] = v