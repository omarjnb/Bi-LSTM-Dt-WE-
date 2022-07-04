# ignore warnings
import warnings
warnings.filterwarnings("ignore")


import os
import json
import pathlib
import joblib
import pandas as pd
from constants import *
from pipeline import pipeline
from neural_network import *
from data_extraction import extract_df_from_xml, preprocess_df
from word_embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import gensim.downloader as api


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


def save_pipeline(path):
    """This function saves the pipeline in pkl format"""
    pipeline_to_persist  = pipeline
    joblib.dump(pipeline_to_persist, SAVE_PIPELINE_PATH)


def load_pipeline(path):
    """This function loads the pipeline from pkl file"""
    if os.path.exists(path):
        pipeline = joblib.load(SAVE_PIPELINE_PATH)
        return pipeline
    else:
        save_pipeline(path)
        pipeline = joblib.load(SAVE_PIPELINE_PATH)
        return pipeline

def split_data(X):
    """This function splits data into train and test set"""
    df_train, df_test = train_test_split(X,
                                        test_size = TEST_SIZE,
                                        train_size = 1 - TEST_SIZE,
                                        random_state = 42)
    return df_train, df_test

def load_data(train_data_file_path):
    """This function loads data from various file formats"""
    train_data_file_path = str(train_data_file_path)
    # .xml files requires more data extraction and extract_df_from_xml has been written in data_extraction.py module
    if train_data_file_path.endswith('.xml'):
        df = extract_df_from_xml(train_data_file_path)
        df = preprocess_df(df)
    elif train_data_file_path.endswith('.csv'):
        df = pd.read_csv(train_data_file_path)
        df = preprocess_df(df)
    elif train_data_file_path.endswith('.json'):
        df = pd.read_json(train_data_file_path)
        df = preprocess_df(df)

    return df


def encode_target_label_aspect(df):
    """This function label encodes the target and saves the encoding in json file so as to use it while predicting"""
    encoder = LabelEncoder()
    encoder.fit(df.aspectType.tolist())

    target_label = encoder.transform(df.aspectType.tolist())
    target_label = target_label.reshape(-1,1)

    encoded_classes = encoder.classes_
    encoded_class_dict = {k:v for k,v in zip(range(len(encoded_classes)),encoded_classes)}
    with open(ENCODED_ASPECT_PATH,'w') as f:
        json.dump(encoded_class_dict,f)


    return target_label

def encode_target_label_category(df):
    """This function one hot encodes the target and saves the encoding in json file so as to use it while predicting"""
    target_label_ = pd.get_dummies(df.aspectCategory)
    target_label = target_label_.values
    target_label = target_label.reshape(-1,5)

    encoded_classes = target_label_.columns
    encoded_class_dict = {k:v for k,v in zip(range(len(encoded_classes)),encoded_classes)}
    with open(ENCODED_CATEGORY_PATH,'w') as f:
        json.dump(encoded_class_dict,f)

    return target_label


"""
GloVe embedding +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

print('Indexing word vectors.')
import numpy as np

embeddings_index = {}
f = open('C:/Users/Owner/OneDrive - Universiti Sains Malaysia/sentiment-analysis/coding 2020/deep neural network/Aspect-Based-Sentiment-Analysis-CNN/glove/glove.6B.300d/glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
MAX_SEQ_LENGTH = 69
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""



def _run_training(data_df,model_name,model_type):

    # splitting the dataset
    df_train,df_test = split_data(data_df)

    # creating word embedding with the help of word_embeddings module
    # Embedding class is written in word_embeddings.py module
    documents = [_text.split() for _text in df_train.text]
    embedding = Embedding()
    w2v_model = embedding.create_w2v_model(documents)

    # tokenize, pad_sequences_ are written in neural_network.py module
    tokenizer, vocab_size = tokenize(df_train.text)
    X_train = pad_sequences_(df_train.text, tokenizer)
    X_test = pad_sequences_(df_test.text, tokenizer)


    GloVe_tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, lower=False)
    GloVe_tokenizer.fit_on_texts(df_train.text)

    # calling encoding function seperately as aspect type requires binary encoding becuase it has just two labels
    # whereas aspect category requires one hot encoding becuase it has mulitple labels
    if model_type=="aspect_type":
        y_train = encode_target_label_aspect(df_train)
        y_test = encode_target_label_aspect(df_test)
    elif model_type == "aspect_category":
        y_train = encode_target_label_category(df_train)
        y_test = encode_target_label_category(df_test)

    embedding_matrix = embedding_matrix_(tokenizer, vocab_size, w2v_model)
    embedding_layer = embedding_layer_(vocab_size , embedding_matrix)
    
    """
    Embedding GloVe++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    print('Preparing embedding matrix.')

    # prepare embedding matrix
    word_index = GloVe_tokenizer.word_index
    nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words + 1, 300))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    #super(TemporalReshape, self).__init__()
    GloVe_embedding_layer = GloVe_embedding_layer_(nb_words+1, embedding_matrix)
          
          
          
          
    print('Embedding Layer set..')
    #glove_model300 = api.load('glove-wiki-gigaword-300')

    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """



    #GloVe_embedding_layer = 

    if model_type == "aspect_type":
        model = create_model_aspect(embedding_layer)
    elif model_type == "aspect_category":
        model = create_model_category(embedding_layer)

    if model_type == "aspect_type":
        compile_model_aspect(model)
    elif model_type == "aspect_category":
        compile_model_category(model)

    callbacks = callbacks_()
    train(X_train, y_train, callbacks, model,model_name)
    #test(X_train, y_train, callbacks, model,model_name)
    print("Model trained and saved")
    score = evaluate(model, X_test, y_test)
    #s_score = evaluate_P_R_F(model, X_test, y_test)
    return  score



def run_training(train_data_file_path):
    print("Training...")

    # load data
    df = load_data(train_data_file_path)
    relevant_cols = ['text','aspectType','aspectCategory']
    df = df[relevant_cols].copy(deep=True)


    # loading pipeline and transforming the data
    pipeline = load_pipeline(SAVE_PIPELINE_PATH)
    df_transformed = pipeline.fit_transform(df)

    # This part trains model for aspect type classification
    # _run_training is the helper function for run_training
    #_run_training(df_transformed,MODEL_ASPECT_TYPE,model_type="aspect_type")
    #####################################################

    # This part trains model for aspect category classification
    df_transformed_ = df_transformed.copy()
    df_implicit = df_transformed[df_transformed['aspectType']=='implicit']


    _run_training(df_implicit,MODEL_CATEGORY,model_type="aspect_category")
    #####################################################
    

if __name__ == '__main__':
    run_training(TRAIN_DATA_FILE_PATH)
