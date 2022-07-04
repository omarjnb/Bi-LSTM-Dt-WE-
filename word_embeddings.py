import gensim
from constants import *

class Embedding:

    def __init__(self):
        pass

    def create_w2v_model(self,documents):
        w2v_model = gensim.models.word2vec.Word2Vec(size = W2V_SIZE,
                                                    window = W2V_WINDOW,
                                                    min_count = W2V_MIN_COUNT,
                                                    workers = W2V_WORKERS)
        w2v_model.build_vocab(documents)
        words = w2v_model.wv.vocab.keys()
        w2v_model.train(documents,
                        total_examples = len(documents),
                        epochs = W2V_EPOCH)
        return w2v_model
        
