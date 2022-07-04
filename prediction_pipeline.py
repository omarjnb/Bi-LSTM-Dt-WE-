# ignore warnings
import warnings
warnings.filterwarnings("ignore")


import json
import pandas as pd
from pipeline import *
from constants import *
from neural_network import *
import tensorflow as tf 
from tensorflow.keras.models import load_model



def fit_data(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    data = pad_sequences(tokenizer.texts_to_sequences(data), maxlen=300)
    return data

def load_json(filepath):
    with open(filepath,"r") as f:
        data = json.load(f)
    return data

def predict(data):

    data = np.array([data])
    df = pd.DataFrame(data,columns=['text'])
    df_transformed = pipeline.fit_transform(df)
    data = df_transformed['text'].values
    data = fit_data(data)
    model_aspect = load_model(MODEL_ASPECT_TYPE)
    model_category = load_model(MODEL_CATEGORY)
    # aspect_type = model_aspect.predict_classes(data)[0]
    encoded_classes_aspect = load_json(ENCODED_ASPECT_PATH)
    print(encoded_classes_aspect)
    print(model_aspect.predict_classes(data)[0])
    pred_aspect_type = encoded_classes_aspect[str(model_aspect.predict_classes(data)[0][0])]
    print("your aspect type: ",pred_aspect_type)
    if pred_aspect_type == "implicit":
        encoded_classes_category = load_json(ENCODED_CATEGORY_PATH)
        pred_aspect_categ = encoded_classes_category[str(model_category.predict_classes(data)[0])]
        print("your aspect category: ",pred_aspect_categ)

def main(CONSOLE_INPUT=True,data=None):
    """This function basically predicts the aspect type and category on the go and

     displays it in consoleif CONSOLE_INPUT=True"""
    if CONSOLE_INPUT:
        while True:
            data = input("Enter your statement: ")
            predict(data)


if __name__ == '__main__':
    main()
