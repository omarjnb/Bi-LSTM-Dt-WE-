# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from constants import *
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM, GlobalMaxPooling1D, Conv2D
from tensorflow.keras.layers import Bidirectional, Masking
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from layers import AttentionWithContext, Addition


# Set log
import logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def tokenize(text_data):
    """This function allows to vectorize a text corpus, by turning each text into either a sequence of integers"""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size

def pad_sequences_(text_data,tokenizer):
    """This function Pads sequences to the same length"""
    padded_sequence = pad_sequences(tokenizer.texts_to_sequences(text_data), maxlen=SEQUENCE_LENGTH)
    return padded_sequence

def embedding_matrix_(tokenizer, vocab_size, w2v_model):
    """This function creates an embedding matrix to be used in creating embedding layer for neural network"""
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    return embedding_matrix


def embedding_layer_(vocab_size,embedding_matrix):

    layer = Embedding(vocab_size,
                      W2V_SIZE,
                      weights=[embedding_matrix],
                      input_length=SEQUENCE_LENGTH,
                      trainable=False)
    return layer

def GloVe_embedding_layer_(vocab_size, embedding_matrix):
    layer = Embedding(vocab_size,
    W2V_SIZE,
    weights=[embedding_matrix],
    input_length=SEQUENCE_LENGTH,
    trainable=False)
    return layer





def create_model_aspect(embedding_layer):
    """This function creates model for aspect type model"""
    model = Sequential()
    model.add(embedding_layer)
    model.add(Masking(mask_value=0.0))
    model.add(Dropout(DROPOUT_RATE_ASPECT))
    model.add(Bidirectional(LSTM(LSTM_UNITS_ASPECT, dropout=DROPOUT_ASPECT, recurrent_dropout=RECURRENT_DROPOUT_ASPECT)))
    model.add(Dense(DENSE_UNITS_ASPECT_MODEL, activation='sigmoid'))
    print(model.summary())
    return model

def create_model_category(embedding_layer):
    """This function creates model for aspect category model"""
    model = Sequential()
    model.add(embedding_layer)
    model.add(Masking(mask_value=0.0))
    model.add(Dropout(DROPOUT_RATE_CATEG))
    model.add(Bidirectional(LSTM(LSTM_UNITS_CATEG, dropout=DROPOUT_CATEG, recurrent_dropout=RECURRENT_DROPOUT_CATEG)))
    #model.add(LSTM(LSTM_UNITS_CATEG, dropout=DROPOUT_CATEG, recurrent_dropout=RECURRENT_DROPOUT_CATEG))
    # model.add(Conv1D(FILTERS,
    #                 KERNEL_SIZE,
    #                 padding='valid',
    #                 activation='relu',
    #                 strides=1))
    #model.add(GlobalMaxPooling1D())
    
    #model.add(AttentionWithContext(embedding_layer))
	#model.add(Addition(embedding_layer))
    
    model.add(Dense(DENSE_UNITS_CATEGORY_MODEL, activation='sigmoid'))
    print(model.summary())
    return model


def compile_model_aspect(model):

    model.compile(loss=LOSS_ASPECT,
              optimizer=OPTIMIZER,
              metrics=METRICS_LIST)


def compile_model_category(model):

    model.compile(loss=LOSS_CATEGORY,
              optimizer=OPTIMIZER,
              metrics=METRICS_LIST)


def callbacks_():

    callbacks = [ ReduceLROnPlateau(monitor=REDUCELR_MONITOR, patience=REDUCELR_PATIENCE,
                                    cooldown=REDUCELR_COOLDOWN),
                  EarlyStopping(monitor=EARLYSTOP_MONITOR, min_delta=EARLYSTOP_MINDELTA,
                                patience=EARLYSTOP_PATIENCE)]
    return callbacks


def train(X_train,y_train, callbacks, model,model_name):

    model.fit(X_train, y_train,
             batch_size=BATCH_SIZE,
             epochs=EPOCHS,
             validation_split=VALIDATION_SPLIT,
             verbose=VERBOSITY,
             callbacks=callbacks)
    model.save(SAVED_MODELS_DIR / model_name)

def test(X_train,y_train, callbacks, model,model_name):

    y_pred = model.predict(X_train[27:])
    print(y_pred)



def evaluate(model, X_test, y_test):

    score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print("AUC:",score[1])
    print("accuracy:",score[2])
    print("Precision:",score[3])
    print("Recall:",score[4])
    #print("F1:",score[5])
    print("LOSS:",score[0])
    print("f1_score", 2 * (score[3]*score[4])/(score[3]+score[4]))
    
    return score



from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import numpy as np
"""
def evaluate_P_R_F(model, X_test, y_test, batch_size=BATCH_SIZE):
    Rec=recall_score(X_test, y_test, average='macro')
    Prec=precision_score(X_test, y_test, average='macro')
    Fscore=f1_score(X_test, y_test, average='macro')
    print(Rec)


    return Rec

"""












"""
Rec=recall_score(X_train,y_train, average='macro')
Prec=precision_score(X_train,y_train, average='macro')
Fscore=f1_score(X_train,y_train, average='macro')
"""