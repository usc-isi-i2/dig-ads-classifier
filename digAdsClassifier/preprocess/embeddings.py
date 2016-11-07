import digAdsClassifier.definitions.embeddings
import digAdsClassifier.definitions.doc_embeddings
import digAdsClassifier.utility.functions
import codecs
import unicodecsv
import numpy as np

def create(embeddings_file):
    embeddings = digAdsClassifier.definitions.embeddings.Embeddings(embeddings_file)
    return embeddings

def get_doc_embeddings(data_file, embeddings, config, limit=None):
    utility.functions.print_name()
    docs = digAdsClassifier.definitions.doc_embeddings.DocEmbeddings()
    with codecs.open(data_file, 'r', 'utf-8') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        for index, row in enumerate(reader):
            #print index
            if(index == 0):
                label_index = row.index(config.get("labelColumn"))
                text_index = row.index(config.get("textColumn"))
            if(index != 0):
                if (limit and index > limit):
                    break
                doc = dict()
                doc['id'] = index
                doc['label'] = int(row[label_index])
                text = row[text_index].lower()

                word_tokens = digAdsClassifier.utility.functions.get_word_tokens(text)

                doc['embedding'] = combine_word_embeddings(word_tokens, embeddings)

                docs.add_to_data(doc)
    return docs

def combine_word_embeddings(word_tokens, embeddings):
    combined_word_embeddings = None
    for word in word_tokens:
        if word in embeddings.dict:
            word_embeddings = embeddings.dict[word]
            if (combined_word_embeddings is None):
                combined_word_embeddings = np.array(word_embeddings)
            else:
                combined_word_embeddings = digAdsClassifier.utility.functions.combine_arrays(combined_word_embeddings, word_embeddings)
    if(combined_word_embeddings is None):
        combined_word_embeddings = np.zeros(len(embeddings.dict.itervalues().next()))
    return combined_word_embeddings