import os
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

# These are the nltk downloads for word preprocessing requirements
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

class RemovePunctuation(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def remove_punctuation(self,row):
        """This function removes all the punctuation from the given row"""
        table = row.maketrans('','',string.punctuation)
        return row.translate(table)

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self.remove_punctuation)
        return X

class Tokenize(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(lambda row:word_tokenize(row))
        return X

class RemoveStopwords(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def remove_stopwords(self,row):
        stop_words = set(stopwords.words('english'))
        filtered_row = []
        for token in row:
            if token not in stop_words:
                filtered_row.append(token)
        return filtered_row

    def transform(self,X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self.remove_stopwords)
        return X

class Lemmatizer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self, X):
        X = X.copy()
        lemmatizer = WordNetLemmatizer()
        for feature in self.variables:
            X[feature] = X[feature].apply(lambda row:[lemmatizer.lemmatize(word) for word in row])
        return X

class JoinTokens(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(lambda row:" ".join(row))
        return X
