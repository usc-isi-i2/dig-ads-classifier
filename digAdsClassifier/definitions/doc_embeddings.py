import codecs
import json

class DocEmbeddings:
    def __init__(self):
        self.data = list()

    def add_to_data(self, doc):
        self.data.append(doc)

    def mark_as_train(self, index):
        self.data[index]['train'] = 1

    def mark_as_test(self, index):
        self.data[index]['train'] = 0

    def is_train(self, index):
        if(self.data[index]['train'] == 1):
            return True
        return False

    def append(self, another_doc):
        self.data = self.data + another_doc.data