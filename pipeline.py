from sklearn.pipeline import Pipeline
import transformers as tfr

variables = ['text']

pipeline = Pipeline([
        ("remove_punctuation", tfr.RemovePunctuation(variables)),
        ("tokenize", tfr.Tokenize(variables)),
        ("remove_stopwords", tfr.RemoveStopwords(variables)),
        ("lemmatizer", tfr.Lemmatizer(variables)),
        ("join_tokens", tfr.JoinTokens(variables))
])